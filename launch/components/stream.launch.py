"""
Usage:
    ros2 launch autodriver_image_object_detection stream.launch.py num_cameras:=2 frame_ids:=["camera_frame_1", "camera_frame_2"] namespaces:=["/camera_1", "/camera_2"] camera_info_files:=["camera_info_1.yaml", "camera_info_2.yaml"]

Todo:
    * setup file, camera camera and rtsp streaming
    * add composition
"""
import os
import json
from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node, SetRemap, PushRosNamespace, SetParametersFromFile, SetParameter
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression, EnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction, OpaqueFunction, SetEnvironmentVariable, LogInfo, TimerAction
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer

def launch_setup(context, *args, **kwargs):
    # Get package directories
    image_detection_dir = get_package_share_directory('autodriver_image_object_detection')

    # Get launch directories
    image_detection_launch_dir = os.path.join(image_detection_dir, 'launch')

    # Declare launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default="False")
    num_cameras = LaunchConfiguration('num_cameras', default=1)
    frame_ids = LaunchConfiguration('frame_ids', default='["camera_frame"]')
    namespaces = LaunchConfiguration('namespaces', default='["/camera"]')
    camera_info_files = LaunchConfiguration('camera_info_files', default='["camera_info.yaml"]')
    stream_sources = LaunchConfiguration('stream_sources')
    stream_types = LaunchConfiguration('stream_types', default='["file"]')
    fps = LaunchConfiguration('fps', default='["30"]')
    widths = LaunchConfiguration('widths', default='["640"]')
    heights = LaunchConfiguration('heights', default='["480"]')
    gscam_config = LaunchConfiguration('gscam_config', default='[""]')
    loop = LaunchConfiguration('loop', default="True")
    sync_sink = LaunchConfiguration('sync_sink', default="True")
    use_gst_timestamps = LaunchConfiguration('use_gst_timestamps', default="True")
    use_sensor_data_qos = LaunchConfiguration('use_sensor_data_qos', default=False)
    image_encoding = LaunchConfiguration('image_encoding', default='rgb8')

    # use_composition = LaunchConfiguration('use_composition', default=False)

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='Use simulation (Gazebo) clock if true')

    declare_num_cameras_cmd = DeclareLaunchArgument(
            'num_cameras',
            default_value=num_cameras,
            description='Number of cameras to launch'
    )

    declare_frame_ids_cmd = DeclareLaunchArgument(
            'frame_ids',
            default_value=frame_ids,
            description='List of frame_ids for each camera'
    )

    declare_namespaces_cmd = DeclareLaunchArgument(
            'namespaces',
            default_value=namespaces,
            description='List of namespaces for each camera'
    )

    declare_camera_info_files_cmd = DeclareLaunchArgument(
            'camera_info_files',
            default_value=camera_info_files,
            description='List of camera_info file paths for each camera'
    )

    declare_stream_sources_cmd = DeclareLaunchArgument(
            'stream_sources',
            default_value=stream_sources,
            description='Path to stream sources. Should be a list of paths. '
                        'For each item, if stream type is a file, pass in the path to the file, '
                        'for live camera, pass in the device e.g /dev/video0, '
                        'for rtsp, pass in the url, e.g rtsp://your_camera_ip:port/stream_path.'
    )

    declare_stream_types_cmd = DeclareLaunchArgument(
            'stream_types',
            default_value=stream_types,
            description='List of stream types for each camera. Options: "file", "rtsp", "camera"'
    )

    declare_fps_cmd = DeclareLaunchArgument(
            'fps',
            default_value=fps,
            description='List of fps for each camera. Can also pass in a single integer that applies to all.'
    )

    declare_widths_cmd = DeclareLaunchArgument(
            'widths',
            default_value=widths,
            description='List of widths for each camera. Can also pass in a single integer that applies to all.'
    )

    declare_heights_cmd = DeclareLaunchArgument(
            'heights',
            default_value=heights,
            description='List of heights for each camera. Can also pass in a single integer that applies to all.'
    )

    declare_gscam_config_cmd = DeclareLaunchArgument(
            'gscam_config',
            default_value=gscam_config,
            description='List of gscam config file paths for each stream. '
                        'Pass in a list of empty strings to infer from stream source.'
    )

    declare_loop_cmd = DeclareLaunchArgument(
            'loop',
            default_value=loop,
            description='Loop file stream if true.'
    )

    declare_sync_sink_cmd = DeclareLaunchArgument(
            'sync_sink',
            default_value=sync_sink,
            description='Synchronize the app sink. '
                        'Sometimes setting this to false can resolve problems with sub-par framerates.'
    )

    declare_use_gst_timestamps_cmd = DeclareLaunchArgument(
            'use_gst_timestamps',
            default_value=use_gst_timestamps,
            description='Use the GStreamer buffer timestamps for the image message header timestamps. '
                        'Setting this to false results in header timestamps being the time that the image buffer transfer is completed.'
    )

    declare_use_sensor_data_qos_cmd = DeclareLaunchArgument(
            'use_sensor_data_qos',
            default_value=use_sensor_data_qos,
            description='The flag to use sensor data qos for camera topic(image, camera_info)'
    )

    declare_image_encoding_cmd = DeclareLaunchArgument(
            'image_encoding',
            default_value=image_encoding,
            description='image encoding ("rgb8", "mono8", "yuv422", "jpeg").'
    )

    # declare_use_composition_cmd = DeclareLaunchArgument(
    #         'use_composition',
    #         default_value=use_composition,
    #         description='Use composition if true'
    # )

    launch_args = [
        declare_use_sim_time_cmd,
        declare_num_cameras_cmd,
        declare_frame_ids_cmd,
        declare_namespaces_cmd,
        declare_camera_info_files_cmd,
        declare_stream_sources_cmd,
        declare_stream_types_cmd,
        declare_fps_cmd,
        declare_widths_cmd,
        declare_heights_cmd,
        declare_gscam_config_cmd,
        declare_loop_cmd,
        declare_sync_sink_cmd,
        declare_use_gst_timestamps_cmd,
        declare_use_sensor_data_qos_cmd,
        declare_image_encoding_cmd,
        # declare_use_composition_cmd,
    ]

    # Launch nodes
    num_cameras_int = int(num_cameras.perform(context))
    frame_ids_list = json.loads(frame_ids.perform(context))  # frame_ids.perform(context).strip('[]').split(',')
    namespaces_list = json.loads(namespaces.perform(context))  # namespaces.perform(context).strip('[]').split(',')
    camera_info_files_list = json.loads(camera_info_files.perform(context))
    stream_sources_list = json.loads(stream_sources.perform(context))
    stream_types_list = json.loads(stream_types.perform(context))
    try:
        fps_list = json.loads(fps.perform(context))
    except (json.JSONDecodeError, json.decoder.JSONDecodeError):
        fps_list = [int(fps.perform(context))]

    try:
        widths_list = json.loads(widths.perform(context))
    except (json.JSONDecodeError, json.decoder.JSONDecodeError):
        widths_list = [int(widths.perform(context))]

    try:
        heights_list = json.loads(heights.perform(context))
    except (json.JSONDecodeError, json.decoder.JSONDecodeError):
        heights_list = [int(heights.perform(context))]

    gscam_config_list = json.loads(gscam_config.perform(context))
    image_encoding = image_encoding.perform(context)

    # Ensure lists have the correct length
    assert len(frame_ids_list) == num_cameras_int, "frame_ids list length must match num_cameras"
    assert len(namespaces_list) == num_cameras_int, "namespaces list length must match num_cameras"
    assert len(camera_info_files_list) == num_cameras_int, "camera_info_files list length must match num_cameras"
    assert len(stream_sources_list) == num_cameras_int, "stream_sources list length must match num_cameras"
    assert len(stream_types_list) == num_cameras_int, "stream_types list length must match num_cameras"
    assert len(gscam_config_list) == num_cameras_int, "gscam_config list length must match num_cameras"

    # Generate gscam nodes
    nodes_to_launch = []

    for i in range(num_cameras_int):
        # If the config is empty, use stream sources to choose
        if gscam_config_list[i] == "":
            format = f'video/x-raw,framerate={fps_list[i]},width={widths_list[i]},height={heights_list[i]}'
            if image_encoding == "jpeg":
                format += ' ! jpegenc ! multipartmux ! multipartdemux ! jpegparse'
            if stream_types_list[i] == "file":
                gscam_config_list[i] = f"filesrc location={stream_sources_list[i]} ! decodebin ! videoconvert ! {format}"
            elif stream_types_list[i] == "rtsp":
                gscam_config_list[i] = f"rtspsrc location={stream_sources_list[i]} latency=50 ! decodebin ! videoconvert ! x264enc ! mp4mux sync=false"
            elif stream_types_list[i] == "camera":
                gscam_config_list[i] = f"v4l2src do-timestamp=true !  device={stream_sources_list[i]} ! {format} ! videoconvert"

        # Generate gscam node
        gscam_node = Node(
            package='gscam',
            executable='gscam_node',
            name='gscam_' + str(i),
            namespace=namespaces_list[i].strip(),
            output='screen',
            parameters=[{
                'frame_id': frame_ids_list[i],
                'camera_info_url': 'file://' + camera_info_files_list[i].strip(),
                'use_sim_time': use_sim_time,
                'gscam_config': gscam_config_list[i],
                'reopen_on_eof': loop,
                'sync_sink': sync_sink,
                'use_gst_timestamps': use_gst_timestamps,
                'use_sensor_data_qos': use_sensor_data_qos,
                'image_encoding': image_encoding
            }],
        )
        nodes_to_launch.append(gscam_node)

    # # Generate autodriver_image_object_detection nodes
    # for i in range(num_cameras_int):
    #     nodes_to_launch += [{
    #         'package': 'autodriver_image_object_detection',
    #         'executable': 'yolo_detector',  # yolo_detection_node
    #         'name': 'yolo_detection_node_' + str(i),
    #         'namespace': namespaces_list[i],
    #         'output': 'screen',
    #         'parameters': [{
    #             'use_sim_time': use_sim_time,
    #             'frame_id': frame_ids_list[i],
    #             'num_cameras': num_cameras,
    #             'namespaces': namespaces,
    #             'camera_info_files': camera_info_files,
    #         }],
    #     }]

    # return the launch description
    ld = launch_args + nodes_to_launch
    return ld


def generate_launch_description():
    return LaunchDescription(
            [
                SetEnvironmentVariable(name='RCUTILS_COLORIZED_OUTPUT', value='1'),
                OpaqueFunction(function=launch_setup)
            ]
    )
