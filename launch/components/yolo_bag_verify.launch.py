"""
Replay a rosbag through yolo_detector (the full sensor-fusion node).

bag_test.launch.py covers single_stream_detector + depth_fusion_node and does NOT
exercise yolo_detection_node's own synchronizer, TF caches or projection paths.
This file does, so changes to that node can be A/B'd against a recorded bag.

Usage:
    ros2 launch autodriver_image_object_detection yolo_bag_verify.launch.py \
        bag_path:=${HOME}/bags

Args:
    bag_path            Path to the .mcap bag directory (required)
    model_path          YOLO model file. Default: yolo11n-seg.pt from the package share dir.
                        Use .pt on desktop/WSL — the .engine files in the repo are TensorRT
                        builds compiled for Jetson and will not load on an RTX GPU.
    output_frame        TF frame for 3D detections. Default: sensor_kit_link (the common
                        ancestor of the gosling1 camera frames, reachable without base_link).
    use_depth           Depth-image projection path (default: true)
    use_pointcloud      Pointcloud/DBSCAN projection path (default: true, needs Open3D)
    conf_threshold      YOLO confidence threshold (default: 0.3)
    sync_slop           ApproximateTimeSynchronizer slop, seconds (default: 0.2).
                        color ~6 Hz vs aligned_depth ~7.5 Hz needs >= 0.167 s to match.
    sync_queue_size     Synchronizer queue depth per topic (default: 20). Must be > 1.
    playback_rate       Bag playback rate multiplier (default: 1.0)
    loop_bag            Loop the bag continuously (default: false — an A/B run wants a
                        finite, repeatable pass over the data)
    startup_delay       Seconds before bag playback starts (default: 12.0; the fusion node
                        loads the model and both projection backends)
    play_bag            Launch `ros2 bag play` automatically (default: true)
    use_rviz            Launch RViz2 (default: false)

Topic mapping (gosling1 bag -> yolo_detector):
    /gosling1/camera/color/image_raw                     -> input_image_topic
    /gosling1/camera/color/camera_info                   -> input_camera_info_topic
    /gosling1/camera/aligned_depth_to_color/image_raw    -> depth_image_topic
    /gosling1/camera/aligned_depth_to_color/camera_info  -> depth_camera_info_topic
    /gosling1/camera/downsampled_cloud_from_depth        -> pointcloud_topic
    /gosling1/tf + /gosling1/tf_static                   -> remapped to /tf + /tf_static

Output topics:
    /yolo/detection_results                 Detection2DArray (pixels)
    /yolo/detection3d_depth_results         Detection3DArray (depth path)
    /yolo/detection3d_pointcloud_results    Detection3DArray (pointcloud path)
    /yolo/objects, /yolo/obstacles          metric object arrays
"""

import glob
import math
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, ExecuteProcess, LogInfo, OpaqueFunction, TimerAction,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, SetParameter

_PI_2 = str(-math.pi / 2)

# Nominal D435i internal extrinsics (realsense2_description, use_nominal_extrinsics=true).
# The bag publishes most of these already; these are fallbacks so the TF tree stays
# connected. Duplicates are harmless — TF2 uses the latest value.
# parent, child, x, y, z, roll, pitch, yaw
_D435I_STATIC_TFS = [
    ('camera_link',         'camera_depth_frame',          '0', '0',      '0', '0',   '0', '0'),
    ('camera_depth_frame',  'camera_depth_optical_frame',  '0', '0',      '0', _PI_2, '0', _PI_2),
    ('camera_link',         'camera_infra1_frame',         '0', '0',      '0', '0',   '0', '0'),
    ('camera_infra1_frame', 'camera_infra1_optical_frame', '0', '0',      '0', _PI_2, '0', _PI_2),
    ('camera_link',         'camera_infra2_frame',         '0', '-0.050', '0', '0',   '0', '0'),
    ('camera_infra2_frame', 'camera_infra2_optical_frame', '0', '0',      '0', _PI_2, '0', _PI_2),
    ('camera_link',         'camera_color_frame',          '0', '0.015',  '0', '0',   '0', '0'),
    ('camera_color_frame',  'camera_color_optical_frame',  '0', '0',      '0', _PI_2, '0', _PI_2),
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


def _node_env():
    """PYTHONPATH for the node's child process.

    ROS 2 console entry points use a #!/usr/bin/python3 shebang and so miss the
    venv's torch/ultralytics/open3d. Derive site-packages from VIRTUAL_ENV (and
    sys.path, for editable/non-venv setups) and inject it via additional_env.
    """
    existing = os.environ.get('PYTHONPATH', '')
    existing_set = set(existing.split(':')) if existing else set()

    extra = []
    venv_root = os.environ.get('VIRTUAL_ENV', '')
    if venv_root:
        for d in sorted(glob.glob(os.path.join(venv_root, 'lib', 'python3*', 'site-packages'))):
            if d not in existing_set:
                extra.append(d)
    for p in sys.path:
        if 'site-packages' in p and p not in existing_set and p not in extra:
            extra.append(p)

    if extra:
        pythonpath = ':'.join(extra) + (':' + existing if existing else '')
    else:
        pythonpath = existing
    return {'PYTHONPATH': pythonpath} if pythonpath else {}


def _resolve_model(pkg, model_path):
    """Resolve a relative model path against the package share dir, then the source tree."""
    if os.path.isabs(model_path):
        return model_path
    resolved = os.path.join(pkg, model_path)
    if os.path.exists(resolved):
        return resolved
    return os.path.normpath(os.path.join(
        pkg, '..', '..', '..', '..', 'src', 'autodriver_image_object_detection',
        'autodriver_image_object_detection', model_path))


def launch_setup(context, *args, **kwargs):
    def cfg(name):
        return LaunchConfiguration(name).perform(context)

    def flag(name):
        return cfg(name).lower() in ('true', '1')

    pkg = get_package_share_directory('autodriver_image_object_detection')
    bag_path = cfg('bag_path')
    startup_delay = float(cfg('startup_delay'))

    actions = [SetParameter(name='use_sim_time', value=True)]
    actions.extend(_d435i_static_tf_nodes())

    actions.append(Node(
        package='autodriver_image_object_detection',
        executable='yolo_detector',
        name='image_obstacle_detection_node',
        output='screen',
        additional_env=_node_env(),
        parameters=[{
            'use_sim_time': True,
            'input_image_topic': '/gosling1/camera/color/image_raw',
            'input_camera_info_topic': '/gosling1/camera/color/camera_info',
            'depth_image_topic': '/gosling1/camera/aligned_depth_to_color/image_raw',
            'depth_camera_info_topic': '/gosling1/camera/aligned_depth_to_color/camera_info',
            'pointcloud_topic': '/gosling1/camera/downsampled_cloud_from_depth',
            'model_path': _resolve_model(pkg, cfg('model_path')),
            'conf_thresh': float(cfg('conf_threshold')),
            'project_to_3d': True,
            'use_depth': flag('use_depth'),
            'use_pointcloud': flag('use_pointcloud'),
            'output_frame': cfg('output_frame'),
            'depth_scale': 1000.0,   # RealSense D435i: 16UC1 mm -> m
            'depth_max': 10.0,       # indoor/near-field cap
            'synchronization_interval': float(cfg('sync_slop')),
            'queue_size': int(cfg('sync_queue_size')),
            'qos': 'SENSOR_DATA',
            'static_camera_info': True,
            'static_camera_to_robot_tf': True,
            'show_image': False,
            'track_2d': flag('track_2d'),
        }],
    ))

    if flag('use_rviz'):
        rviz_config = os.path.join(pkg, 'config', 'bag_test.rviz')
        actions.append(Node(
            package='rviz2', executable='rviz2', name='rviz2', output='screen',
            arguments=['-d', rviz_config] if os.path.exists(rviz_config) else [],
            parameters=[{'use_sim_time': True}],
        ))

    if flag('play_bag'):
        if not bag_path:
            actions.append(LogInfo(
                msg='[yolo_bag_verify] play_bag=true but bag_path is empty — skipping playback'))
        else:
            bag_cmd = [
                'ros2', 'bag', 'play', bag_path,
                '--clock',
                '--rate', cfg('playback_rate'),
                '--read-ahead-queue-size', '1000',
                '--remap', '/gosling1/tf:=/tf',
                '--remap', '/gosling1/tf_static:=/tf_static',
            ]
            if flag('loop_bag'):
                bag_cmd.append('--loop')
            actions.append(LogInfo(
                msg=f'[yolo_bag_verify] Bag play starts in {startup_delay:.0f}s: {bag_path}'))
            actions.append(TimerAction(
                period=startup_delay,
                actions=[ExecuteProcess(cmd=bag_cmd, output='screen')],
            ))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('bag_path', default_value='',
                              description='Path to the rosbag directory'),
        DeclareLaunchArgument('model_path', default_value='yolo11n-seg.pt',
                              description='YOLO model. Use .pt on desktop/WSL, .engine on Jetson'),
        DeclareLaunchArgument('output_frame', default_value='sensor_kit_link',
                              description='TF frame for 3D output; empty to use the sensor frame'),
        DeclareLaunchArgument('use_depth', default_value='true',
                              description='Enable the depth-image projection path'),
        DeclareLaunchArgument('use_pointcloud', default_value='true',
                              description='Enable the pointcloud/DBSCAN projection path'),
        DeclareLaunchArgument('conf_threshold', default_value='0.3',
                              description='YOLO confidence threshold'),
        DeclareLaunchArgument('track_2d', default_value='true',
                              description='2D tracking. Set false for an A/B run: model.track '
                                          'carries state across frames, so two runs that process '
                                          'even slightly different frame sets diverge everywhere. '
                                          'With tracking off each frame is independent and matched '
                                          'timestamps are directly comparable.'),
        DeclareLaunchArgument('sync_slop', default_value='0.2',
                              description='Synchronizer slop (s); needs >= 0.167 for this bag'),
        DeclareLaunchArgument('sync_queue_size', default_value='20',
                              description='Synchronizer queue depth; must be > 1 for multi-rate topics'),
        DeclareLaunchArgument('playback_rate', default_value='1.0',
                              description='Bag playback rate multiplier'),
        DeclareLaunchArgument('loop_bag', default_value='false',
                              description='Loop the bag; keep false for a repeatable A/B pass'),
        DeclareLaunchArgument('startup_delay', default_value='12.0',
                              description='Seconds before bag playback starts'),
        DeclareLaunchArgument('play_bag', default_value='true',
                              description='Launch `ros2 bag play` automatically'),
        DeclareLaunchArgument('use_rviz', default_value='false',
                              description='Launch RViz2'),
        OpaqueFunction(function=launch_setup),
    ])
