"""
Bag test launch: play a ROSBAG and run single_stream_detector + depth_fusion_node.

Usage:
    ros2 launch autodriver_image_object_detection bag_test.launch.py \
        bag_path:=/mnt/d/Coding/Projects/f1tenth/test_bags_05252026/loop3x_with_localization_and_pointcloud

Args:
    bag_path            Path to the .mcap bag directory (required)
    model_path          YOLO model file. Default: yolo11n-seg.pt.
                        Use .pt files for both WSL GPU (RTX) and CPU — the .engine files in the
                        repo are TensorRT builds compiled for Jetson and will not run on an RTX GPU.
    output_frame        TF frame for 3D detections. Default: '' (use source camera frame;
                        avoids TF lookup failure when base_link is absent from the bag)
    use_depth           Use aligned depth image for 3D projection (default: true)
    use_pointcloud      Use downsampled pointcloud for 3D projection (default: true)
                        Requires Open3D CUDA. color (~6 Hz) and downsampled_cloud (~19 Hz) have
                        different timestamps; sync_slop and sync_queue_size handle the mismatch.
    play_bag            Launch ros2 bag play automatically (default: true)
    playback_rate       Bag playback rate multiplier (default: 1.0).
                        GPU inference is fast enough; color feed is only ~6 Hz so node is input-limited.
    loop_bag            Loop the bag continuously (default: true — avoids bag finishing mid-test)
    startup_delay       Seconds to wait before starting bag playback (default: 8.0).
                        GPU model load + fuse is faster than CPU; 8 s is sufficient.
    use_rviz            Launch RViz2 (default: false)
    conf_threshold      YOLO confidence threshold (default: 0.3)
    sync_slop           ApproximateTimeSynchronizer slop in seconds (default: 0.2).
                        color at ~6 Hz and aligned_depth at ~7.5 Hz need ≥ 0.167 s to reliably match.
    sync_queue_size     Synchronizer queue depth per topic (default: 20).
                        Must be > 1 to buffer messages across different-rate topics.

Topic mapping (gosling1 bag → nodes):
    /gosling1/camera/color/image_raw                      → single_stream_detector input
    /gosling1/camera/color/camera_info                    → single_stream_detector + depth_fusion rgb_info
    /gosling1/camera/aligned_depth_to_color/image_raw     → depth_fusion depth input
    /gosling1/camera/aligned_depth_to_color/camera_info   → depth_fusion depth_info
    /gosling1/camera/downsampled_cloud_from_depth         → depth_fusion pointcloud input (optional)
    /gosling1/tf + /gosling1/tf_static                    → remapped to /tf + /tf_static via bag play

Detection output topics:
    /gosling1/yolo/detection_results      Detection2DArray
    /gosling1/yolo/detection_image        Annotated RGB image
    /depth_fusion/detection3d_depth       Detection3DArray (depth path)
    /depth_fusion/detection3d_pointcloud  Detection3DArray (pointcloud path, if use_pointcloud=true)
"""

import glob
import os
import sys
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

import math
_PI_2 = str(-math.pi / 2)

# Nominal D435i extrinsics from realsense2_description/_d435.urdf.xacro
# (use_nominal_extrinsics=true).  These cover frames the bag may not publish
# (e.g. infra1/infra2/depth) so the TF tree is fully connected.
# parent, child, x, y, z, roll, pitch, yaw
_D435I_STATIC_TFS = [
    # Nominal D435i internal extrinsics (realsense2_description use_nominal_extrinsics=true).
    # The bag already publishes most of these via /gosling1/tf_static, so these are fallbacks.
    # NOTE: the bag uses camera_aligned_depth_to_infra1_frame as parent of
    # camera_infra1_optical_frame (not camera_infra1_frame). TF2 will use whichever is
    # published last (bag wins after the startup_delay, which is fine).
    ('camera_link',                'camera_depth_frame',            '0',      '0',       '0',      '0',   '0',   '0'),
    ('camera_depth_frame',         'camera_depth_optical_frame',    '0',      '0',       '0',      _PI_2, '0',   _PI_2),
    ('camera_link',                'camera_infra1_frame',           '0',      '0',       '0',      '0',   '0',   '0'),
    ('camera_infra1_frame',        'camera_infra1_optical_frame',   '0',      '0',       '0',      _PI_2, '0',   _PI_2),
    ('camera_link',                'camera_infra2_frame',           '0',      '-0.050',  '0',      '0',   '0',   '0'),
    ('camera_infra2_frame',        'camera_infra2_optical_frame',   '0',      '0',       '0',      _PI_2, '0',   _PI_2),
    ('camera_link',                'camera_color_frame',            '0',      '0.015',   '0',      '0',   '0',   '0'),
    ('camera_color_frame',         'camera_color_optical_frame',    '0',      '0',       '0',      _PI_2, '0',   _PI_2),
]


def _d435i_static_tf_nodes():
    """Return one static_transform_publisher Node per D435i nominal extrinsic."""
    nodes = []
    for parent, child, x, y, z, roll, pitch, yaw in _D435I_STATIC_TFS:
        nodes.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'tf_{child.replace("/", "_")}',
            arguments=[
                '--x', x, '--y', y, '--z', z,
                '--roll', roll, '--pitch', pitch, '--yaw', yaw,
                '--frame-id', parent,
                '--child-frame-id', child,
            ],
            parameters=[{'use_sim_time': True}],
            output='screen',
        ))
    return nodes


def launch_setup(context, *args, **kwargs):
    pkg = get_package_share_directory('autodriver_image_object_detection')

    bag_path = LaunchConfiguration('bag_path').perform(context)
    model_path = LaunchConfiguration('model_path').perform(context)
    output_frame = LaunchConfiguration('output_frame').perform(context)
    use_depth = LaunchConfiguration('use_depth').perform(context).lower() in ('true', '1')
    use_pointcloud = LaunchConfiguration('use_pointcloud').perform(context).lower() in ('true', '1')
    play_bag = LaunchConfiguration('play_bag').perform(context).lower() in ('true', '1')
    playback_rate = LaunchConfiguration('playback_rate').perform(context)
    loop_bag = LaunchConfiguration('loop_bag').perform(context).lower() in ('true', '1')
    startup_delay = float(LaunchConfiguration('startup_delay').perform(context))
    use_rviz = LaunchConfiguration('use_rviz').perform(context).lower() in ('true', '1')
    conf_threshold = LaunchConfiguration('conf_threshold').perform(context)
    sync_slop = float(LaunchConfiguration('sync_slop').perform(context))
    sync_queue_size = int(LaunchConfiguration('sync_queue_size').perform(context))
    publish_empty_detections = (
        LaunchConfiguration('publish_empty_detections').perform(context).lower() in ('true', '1'))

    # Per-stage profiling params (shared by both detector nodes). Off by default.
    profile_params = {
        'profile': LaunchConfiguration('profile').perform(context).lower() in ('true', '1'),
        'profile_log': LaunchConfiguration('profile_log').perform(context).lower() in ('true', '1'),
        'profile_publish': (
            LaunchConfiguration('profile_publish').perform(context).lower() in ('true', '1')),
        'profile_log_interval': float(
            LaunchConfiguration('profile_log_interval').perform(context)),
        'profile_window': int(LaunchConfiguration('profile_window').perform(context)),
    }

    # Resolve model path: if not absolute, look in package share directory
    if not os.path.isabs(model_path):
        resolved_model = os.path.join(pkg, model_path)
        if not os.path.exists(resolved_model):
            src_model = os.path.normpath(os.path.join(
                pkg, '..', '..', '..', '..', 'src',
                'autodriver_image_object_detection',
                'autodriver_image_object_detection', model_path
            ))
            resolved_model = src_model
    else:
        resolved_model = model_path

    # --- PYTHONPATH for node child processes ---
    # Entry-point scripts use #!/usr/bin/python3 (system Python shebang) and don't see
    # venv packages. We derive venv site-packages from VIRTUAL_ENV (set by venv activate)
    # and inject via additional_env. This works even if ros2 itself runs system Python,
    # because VIRTUAL_ENV is inherited from the shell environment.
    existing_pp = os.environ.get('PYTHONPATH', '')
    existing_pp_set = set(existing_pp.split(':')) if existing_pp else set()

    extra = []
    venv_root = os.environ.get('VIRTUAL_ENV', '')
    if venv_root:
        for d in sorted(glob.glob(os.path.join(venv_root, 'lib', 'python3*', 'site-packages'))):
            if d not in existing_pp_set:
                extra.append(d)
    # Also include sys.path site-packages (catches editable installs / non-venv setups)
    for p in sys.path:
        if 'site-packages' in p and p not in existing_pp_set and p not in extra:
            extra.append(p)

    if extra:
        node_pythonpath = ':'.join(extra) + (':' + existing_pp if existing_pp else '')
    else:
        node_pythonpath = existing_pp
    node_env = {'PYTHONPATH': node_pythonpath} if node_pythonpath else {}

    actions = []

    # sim time must be set before any node starts
    actions.append(SetParameter(name='use_sim_time', value=True))

    # --- D435i nominal static TFs ---
    # Supplement the bag's TF tree with the full internal camera chain so frames
    # like camera_infra1_optical_frame (pointcloud) are connected to camera_link.
    # Duplicates of frames already in the bag are harmless — TF2 uses the latest value.
    actions.extend(_d435i_static_tf_nodes())

    # --- 2D detector ---
    actions.append(Node(
        package='autodriver_image_object_detection',
        executable='single_stream_detector',
        name='single_stream_detector',
        output='screen',
        additional_env=node_env,
        parameters=[{
            'use_sim_time': True,
            'input_image_topic': '/gosling1/camera/color/image_raw',
            'input_camera_info_topic': '/gosling1/camera/color/camera_info',
            'detection_results_topic': 'gosling1/yolo/detection_results',
            'detection_image_topic': 'gosling1/yolo/detection_image',
            'segmentation_image_topic': 'gosling1/yolo/segmentation_image',
            'segmentation_mask_image_topic': 'gosling1/yolo/segmentation_mask_image',
            'model_path': resolved_model,
            'conf_thresh': float(conf_threshold),
            'track_2d': True,
            'show_image': False,
            'qos': 'SENSOR_DATA',
            **profile_params,
        }],
    ))

    # --- 3D depth fusion ---
    # sync_slop and sync_queue_size fix:
    #   color (~6 Hz) and aligned_depth (~7.5 Hz) have different timestamps —
    #   slop must be ≥ 1/min_rate = 0.167 s to reliably match across one cycle.
    #   queue_size=1 (old default) almost never fires with multi-rate topics; use 20.
    actions.append(Node(
        package='autodriver_image_object_detection',
        executable='depth_fusion_node',
        name='depth_fusion_node',
        output='screen',
        additional_env=node_env,
        parameters=[{
            'use_sim_time': True,
            'detections_2d_topic': 'gosling1/yolo/detection_results',
            'rgb_camera_info_topic': '/gosling1/camera/color/camera_info',
            'depth_image_topic': '/gosling1/camera/aligned_depth_to_color/image_raw',
            'depth_camera_info_topic': '/gosling1/camera/aligned_depth_to_color/camera_info',
            'pointcloud_topic': '/gosling1/camera/downsampled_cloud_from_depth',
            'output_frame': output_frame,
            'use_depth': use_depth,
            'use_pointcloud': use_pointcloud,
            'depth_scale': 1000.0,   # RealSense D435i: 16UC1 mm → m
            'depth_max': 10.0,       # cap at 10 m for indoor/near-field
            'synchronization_interval': sync_slop,
            'queue_size': sync_queue_size,
            'qos': 'SENSOR_DATA',
            'static_camera_info': True,
            'static_camera_to_robot_tf': True,
            'publish_empty_detections': publish_empty_detections,
            **profile_params,
        }],
    ))

    # --- RViz2 ---
    if use_rviz:
        rviz_config = os.path.join(pkg, 'config', 'bag_test.rviz')
        rviz_args = ['-d', rviz_config] if os.path.exists(rviz_config) else []
        actions.append(Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=rviz_args,
            parameters=[{'use_sim_time': True}],
        ))

    # --- bag playback (delayed to let nodes finish loading) ---
    if play_bag:
        if not bag_path:
            actions.append(LogInfo(
                msg='[bag_test] play_bag=true but bag_path is empty — skipping playback'))
        else:
            bag_cmd = [
                'ros2', 'bag', 'play', bag_path,
                '--clock',
                '--rate', playback_rate,
                '--read-ahead-queue-size', '1000',
                # remap gosling1-namespaced TF → global /tf so tf2_ros listeners see it
                '--remap', '/gosling1/tf:=/tf',
                '--remap', '/gosling1/tf_static:=/tf_static',
            ]
            if loop_bag:
                bag_cmd.append('--loop')

            actions.append(LogInfo(
                msg=f'[bag_test] Bag play starts in {startup_delay:.0f}s '
                    f'(rate={playback_rate}x, loop={loop_bag}): {bag_path}'))
            actions.append(TimerAction(
                period=startup_delay,
                actions=[ExecuteProcess(cmd=bag_cmd, output='screen')],
            ))

    return actions


def generate_launch_description():
    pkg = get_package_share_directory('autodriver_image_object_detection')
    default_model = os.path.join(pkg, 'yolo11x-seg.pt')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bag_path',
            default_value='',
            description='Path to the rosbag directory',
        ),
        DeclareLaunchArgument(
            'model_path',
            default_value=default_model,
            description='YOLO model. Use .pt for CPU/WSL, .engine on Jetson',
        ),
        DeclareLaunchArgument(
            'output_frame',
            default_value='sensor_kit_link',
            description=(
                'TF frame for 3D detection output. '
                'For this bag: sensor_kit_link is the common ancestor of all camera frames '
                'and is directly reachable via TF without needing base_link. '
                'Use empty string to skip TF and publish in the source camera frame.'
            ),
        ),
        DeclareLaunchArgument(
            'use_depth',
            default_value='true',
            description='Use aligned depth image for 3D projection',
        ),
        DeclareLaunchArgument(
            'use_pointcloud',
            default_value='true',
            description='Use downsampled pointcloud for 3D projection (requires Open3D CUDA)',
        ),
        DeclareLaunchArgument(
            'play_bag',
            default_value='true',
            description='Launch ros2 bag play automatically',
        ),
        DeclareLaunchArgument(
            'playback_rate',
            default_value='1.0',
            description='Bag playback rate (1.0 with GPU; color feed is ~6 Hz so node is input-limited)',
        ),
        DeclareLaunchArgument(
            'loop_bag',
            default_value='true',
            description='Loop the bag continuously',
        ),
        DeclareLaunchArgument(
            'startup_delay',
            default_value='8.0',
            description='Seconds before bag play starts. GPU model load+fuse takes ~5-8 s',
        ),
        DeclareLaunchArgument(
            'use_rviz',
            default_value='false',
            description='Launch RViz2',
        ),
        DeclareLaunchArgument(
            'conf_threshold',
            default_value='0.3',
            description='YOLO confidence threshold',
        ),
        DeclareLaunchArgument(
            'sync_slop',
            default_value='0.2',
            description='ApproximateTimeSynchronizer slop (s). color≈6Hz vs depth≈7.5Hz needs ≥0.167s',
        ),
        DeclareLaunchArgument(
            'sync_queue_size',
            default_value='20',
            description='Synchronizer queue depth per topic. Must be >1 for multi-rate topics',
        ),
        DeclareLaunchArgument(
            'publish_empty_detections',
            default_value='true',
            description='3D nodes: on zero-detection frames publish empty Detection3DArray + '
                        'DELETEALL markers (heartbeat for costmap/marker clearing)',
        ),
        DeclareLaunchArgument(
            'profile',
            default_value='false',
            description='Enable per-stage execution-time profiling (detection/depth/pointcloud)',
        ),
        DeclareLaunchArgument(
            'profile_log',
            default_value='true',
            description='Profiling: emit throttled mean/min/max/count log lines',
        ),
        DeclareLaunchArgument(
            'profile_publish',
            default_value='false',
            description='Profiling: publish per-frame Float32MultiArray on <node>/profile',
        ),
        DeclareLaunchArgument(
            'profile_log_interval',
            default_value='5.0',
            description='Profiling: seconds between aggregate log lines',
        ),
        DeclareLaunchArgument(
            'profile_window',
            default_value='200',
            description='Profiling: rolling-window sample count for aggregate stats',
        ),
        OpaqueFunction(function=launch_setup),
    ])
