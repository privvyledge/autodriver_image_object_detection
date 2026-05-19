"""
Usage:
    ros2 launch autodriver_image_object_detection stream.launch.py num_cameras:=2 frame_ids:=["camera_frame_1", "camera_frame_2"] namespaces:=["/camera_1", "/camera_2"] camera_info_files:=["camera_info_1.yaml", "camera_info_2.yaml"]

Tips: todo: move to a github gist and blog article
    * Sample gstreamer configs
        * Launching a feed
            * File
            * Camera (live feed)
            * RTSP: rtsp://your_camera_ip:port/stream_path
                * Server (optional: if not running elsewhere). Use one:
                    * UDPSink:
                        * Camera
                            * H264:
                                * gst-launch-1.0 -v v4l2src device=/dev/video0 ! decodebin ! x264enc ! rtph264pay ! udpsink host=${IP_ADDRESS} port=${PORT}
                                gst-launch-1.0 -vvvv v4l2src ! 'video/x-raw, width=640, height=480, framerate=30/1' ! videoconvert !  x264enc pass=qual quantizer=20 tune=zerolatency ! rtph264pay ! udpsink port=${PORT}
                        * File (use one):
                            * H264:
                                * gst-launch-1.0 -v filesrc location=${PATH_TO_FILE} ! decodebin ! x264enc ! rtph264pay ! udpsink host=127.0.0.1 port=${PORT}
                                * gst-launch-1.0 -v filesrc location=${PATH_TO_FILE} ! decodebin ! x264enc ! rtph264pay config-interval=1 pt=96 ! udpsink host=${IP_ADDRESS} port=${PORT}
                                * gst-launch-1.0 filesrc location=${PATH_TO_FILE} ! qtdemux ! queue ! h264parse ! rtph264pay config-interval=10 ! udpsink host=${IP_ADDRESS} port=${PORT} -v
                            * JPEG:
                                * gst-launch-1.0 -v filesrc location=${PATH_TO_FILE} ! decodebin ! videoconvert ! jpegenc ! rtpjpegpay ! udpsink host=${IP_ADDRESS} port=${PORT}

                    * TCPSink:
                        * gst-launch-1.0 -vvvv v4l2src ! 'video/x-raw, width=640, height=480, framerate=30/1' ! videoconvert ! jpegenc ! rtpjpegpay ! rtpstreampay ! tcpserversink port=7001

                * Client
                    Test:
                        * gst-launch-1.0 -v videotestsrc ! videoconvert ! videoscale ! video/x-raw,width=640,height=480 ! x264enc speed-preset=veryfast tune=zerolatency bitrate=800 ! rtspclientsink location=rtsp://localhost:8554/test

Todo:
    * add global namespace [done]
    * update the code [done]
    * setup do timestamps [done]
    * Test image encoding [done]
    * make width, height, encoding and fps optional [done]
    * handle single item passed for width, height, fps, etc [done]
    * test camera_info [done]
    * test if compressed topics are published [done]
    * test sensor data qos [done]
    * test using appsink [done]
    * test new list string parsing [done]
    * visualize the image using RQT or RViz [done]
    * add support for file saving (use my package instead) [done]
    * add support for mjpg (H264) encoding and print auto checking result [done]
    * setup file, camera camera and rtsp streaming [done]
    * add support for saving raw image, yolo detection or both [done]
    * add support for ros_deep_learning and jetson_inference for Jetsons (and maybe x86) since gscam doesn't support nvidia Jetsons gstreamer pipelines. [done]
    * add support for model name as an argument
    * replace launch gstreamer config strings with ' ! '.join(list of stuff)
    * add support for automatic stream type inferencing e.g file, rtsp, camera from "stream_sources"
    * move detection to a separate launch file and keep this streaming only
    * add support for disabling detection, gscam or both. Do this by separating launch files.
    * cleanup the autoconfig string addition
    * add composition
"""
import os
import subprocess
import json
import re
import pathlib

from launch import LaunchDescription, LaunchContext
from launch_ros.actions import Node, SetRemap, PushRosNamespace, SetParametersFromFile, SetParameter
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression, EnvironmentVariable
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition, UnlessCondition, LaunchConfigurationEquals, LaunchConfigurationNotEquals
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, GroupAction, OpaqueFunction, \
    SetEnvironmentVariable, LogInfo, TimerAction
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.launch_description_sources import PythonLaunchDescriptionSource, FrontendLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from ament_index_python.packages import PackageNotFoundError as ROS2PackageNotFoundError
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer


def parse_list_string(list_string):
    try:
        parsed_string = json.loads(list_string)
    except json.JSONDecodeError:
        parsed_string = list_string.strip('[]').split(',')
        parsed_string = [s.strip() for s in parsed_string]
    return parsed_string

def post_process_list_string(parsed_string, list_length):
    # todo: refactor this function and simplify
    # todo: handle strings instead of assuming only integers
    post_processed_list = parsed_string
    # handle singular, empty or 'none' value for fps, width, height, encoding, etc.
    # ## single int or float value passed
    if isinstance(parsed_string, (int, float)):
        return [int(parsed_string)] * list_length

    # ## single list value passed
    if len(parsed_string) == 1:
        # handle list of string of length 1, e.g '["1"]'
        if isinstance(parsed_string, str):
            if parsed_string.lower().strip() in ["none", "false"]:
                post_processed_list = [0] * list_length
            else:
                post_processed_list = [int(float(parsed_string))] * list_length
        else:
            post_processed_list = [int(float(parsed_string[0]))] * list_length

    # ## check if a single string is passed and if it is 'none' then set to empty string
    if isinstance(parsed_string, str):
        if parsed_string.lower().strip() in ["none", "false"]:
            post_processed_list = [0] * list_length
        elif parsed_string.lower().strip() == "":
            post_processed_list = [0] * list_length
        else:
            post_processed_list = [int(float(parsed_string))] * list_length

    # ## empty values passed
    if len(parsed_string) == 0:
        post_processed_list = [0] * list_length

    # ## check if a list type passed then assert length matches num_cameras
    if isinstance(parsed_string, list):
        assert len(parsed_string) == list_length, "fps list length must match num_cameras"
        post_processed_list = [int(parsed_string_i) for parsed_string_i in parsed_string]

    # todo: refactor this
    for i in range(list_length):
        if isinstance(parsed_string[i], list) and parsed_string[i] == "":
            post_processed_list[i] = 0
    return post_processed_list

def clean_gstreamer_pipeline(pipeline: str) -> str:
    # Remove appsink from the gsconfig script if present as gscam automatically adds it.
    cleaned_pipeline = re.sub(r'!\s*appsink', '', pipeline).strip()
    return cleaned_pipeline


def launch_setup(context, *args, **kwargs):
    # Define some constants
    mux_types = {
        '.mkv': 'matroskamux',
        '.flv': 'flvmux',
        '.avi': 'avimux',
        '.mp4': 'mp4mux',
        '.mov': 'qtmux',
        '.webm': 'webmmux',
        '.ts': 'tsdemux',
        '.mp2': 'mpegtsmux'
    }

    image_encoding_to_gstreamer_format = {
        'rgb8': 'RGB8',
        'mono8': 'GRAY8',
        'yuv422': 'YUY2',
        'jpeg': ''
    }

    # Get package directories
    image_detection_dir = get_package_share_directory('autodriver_image_object_detection')

    # Get launch directories
    image_detection_launch_dir = os.path.join(image_detection_dir, 'launch')
    image_detection_data_dir = os.path.join(image_detection_dir, 'data')
    image_detection_config_dir = os.path.join(image_detection_dir, 'config')

    # Declare launch configuration variables
    use_sim_time = LaunchConfiguration('use_sim_time', default="False")
    use_global_namespace = LaunchConfiguration('use_global_namespace', default="False")
    global_namespace = LaunchConfiguration('global_namespace', default="/")
    num_cameras = LaunchConfiguration('num_cameras', default=1)
    frame_ids = LaunchConfiguration('frame_ids', default='["camera_frame"]')
    namespaces = LaunchConfiguration('namespaces', default='["/camera"]')
    camera_info_files = LaunchConfiguration('camera_info_files', default='["camera_info.yaml"]')
    stream_sources = LaunchConfiguration('stream_sources')
    stream_format = LaunchConfiguration('stream_format', default='["raw"]')
    stream_types = LaunchConfiguration('stream_types', default='["file"]')
    fps = LaunchConfiguration('fps', default='["30"]')
    widths = LaunchConfiguration('widths', default='640')
    heights = LaunchConfiguration('heights', default='"480"')
    gscam_config = LaunchConfiguration('gscam_config', default='[""]')
    save_videos = LaunchConfiguration('save_videos', default="False")
    video_save_filenames = LaunchConfiguration('video_save_filenames', default='["video0.mp4"]')
    video_save_source = LaunchConfiguration('video_save_source', default='raw')  # raw, detection, both
    video_save_package = LaunchConfiguration('video_save_package', default='custom')  # custom (mine), jetson_inference
    loop = LaunchConfiguration('loop', default="True")
    sync_sink = LaunchConfiguration('sync_sink', default="True")
    use_gst_timestamps = LaunchConfiguration('use_gst_timestamps', default="True")
    use_sensor_data_qos = LaunchConfiguration('use_sensor_data_qos', default=False)
    image_encoding = LaunchConfiguration('image_encoding', default='rgb8')
    stream_package = LaunchConfiguration('stream_package', default='jetson_inference')  # gscam, jetson_inference

    # use_composition = LaunchConfiguration('use_composition', default=False)

    # Declare launch arguments
    declare_use_sim_time_cmd = DeclareLaunchArgument(
            'use_sim_time',
            default_value=use_sim_time,
            description='Use simulation (Gazebo) clock if true')

    declare_use_global_namespace_cmd = DeclareLaunchArgument(
            'use_global_namespace',
            default_value=use_global_namespace,
            description='Prepend a global namespace if true.'
    )

    declare_global_namespace_cmd = DeclareLaunchArgument(
            'global_namespace',
            default_value=global_namespace,
            description='Global namespace prepended to all nodes generated.'
    )

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
                        'for rtsp, pass in the url, e.g rtsp://your_camera_ip:port/stream_path. '
                        'Used only if gscam_config is empty.'
    )

    declare_stream_format_cmd = DeclareLaunchArgument(
            'stream_format',
            default_value=stream_format,
            description='List of stream formats for each camera. Options: "raw", "mjpg". '
                        'Make sure to check the formats available for your stream_type(s) '
                        'by running: "v4l2-ctl --list-formats-ext". Usually YUYV and MJPG are available. '
                        'This was mostly added to handle MJPG streams correctly since the GSCAM node does not publish '
                        'MJPGs correctly, e.g if high resolution/fps is passed even though the camera supports it. '
                        'Tip: specify "MJPG" and image_encoding:=rgb8 to get high rate MJPGs'
                        'Used only if gscam_config is empty.'
    )

    declare_stream_types_cmd = DeclareLaunchArgument(
            'stream_types',
            default_value=stream_types,
            description='List of stream types for each camera. Options: "file", "rtsp", "camera". '
                        'Used only if gscam_config is empty.'
    )

    declare_fps_cmd = DeclareLaunchArgument(
            'fps',
            default_value=fps,
            description='List of fps for each camera. Can also pass in a single integer that applies to all. '
                        'Set to 0 to use the default FPS. '
                        'If Gstreamer fails to get a sample, '
                        'it is probably due to passing in unsupported width+height+fps+encoding combinations. '
                        'Run "v4l2-ctl --list-formats-ext" to see the supported combinations for the cameras.'
                        'Used only if gscam_config is empty.'
    )

    declare_widths_cmd = DeclareLaunchArgument(
            'widths',
            default_value=widths,
            description='List of widths for each camera. Can also pass in a single integer that applies to all. '
                        'Set to 0 to use the default FPS. '
                        'If Gstreamer fails to get a sample, '
                        'it is probably due to passing in unsupported width+height+fps+encoding combinations. '
                        'Run "v4l2-ctl --list-formats-ext" to see the supported combinations for the cameras.'
                        'Used only if gscam_config is empty.'
    )

    declare_heights_cmd = DeclareLaunchArgument(
            'heights',
            default_value=heights,
            description='List of heights for each camera. Can also pass in a single integer that applies to all. '
                        'Set to 0 to use the default FPS. '
                        'If Gstreamer fails to get a sample, '
                        'it is probably due to passing in unsupported width+height+fps+encoding combinations. '
                        'Run "v4l2-ctl --list-formats-ext" to see the supported combinations for the cameras.'
                        'Used only if gscam_config is empty.'
    )

    declare_gscam_config_cmd = DeclareLaunchArgument(
            'gscam_config',
            default_value=gscam_config,
            description='List of gscam config file paths for each stream. '
                        'Pass in a list of empty strings to infer from stream source.'
    )

    declare_save_videos_cmd = DeclareLaunchArgument(
            'save_videos',
            default_value=save_videos,
            description='Save videos if true.'
    )

    declare_video_save_filenames_cmd = DeclareLaunchArgument(
            'video_save_filenames',
            default_value=video_save_filenames,
            description='List of video save filenames for each camera. '
                        'Used only if save_videos is true. Recommended: .mp4, .avi, .mpeg'
    )

    declare_video_save_source_cmd = DeclareLaunchArgument(
            'video_save_source',
            default_value=video_save_source,
            description='The image topic to use when saving videos. Options: raw, detection, both'
    )

    declare_video_save_package_cmd = DeclareLaunchArgument(
        'video_save_package',
        default_value=video_save_package,
        description='The package to use to save videos. Options: mine (supports static FPS), jetson_inference (better).'
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
            description='image encoding ("rgb8", "mono8", "yuv422", "jpeg"). '
                        'The image encoding used by the ROS2 publisher. '
                        'This is usually different from the stream format '
                        'although its usually the same except for jpeg. '
                        'Recommended: do not use "jpeg" as the '
                        ' image driver automatically publishes compressed jpegs for the other encodings.'
                        'If using jpeg, publishes only a compressed image, i.e appends "/compressed to the topic name.".'
                        '"jpeg" may be deprecated in future versions of this launch files.'
    )

    declare_stream_package_cmd = DeclareLaunchArgument(
        'stream_package',
        default_value=stream_package,
        description='The ROS package to use to open streams. Either "gscam" or "jetson_inference"'
    )

    declare_model_path_cmd = DeclareLaunchArgument(
        'model_path',
        default_value=os.path.join(image_detection_dir, 'yolo11x.engine'),
        description='Path to the YOLO model file (.engine, .pt, .onnx) used by detection nodes'
    )

    # declare_use_composition_cmd = DeclareLaunchArgument(
    #         'use_composition',
    #         default_value=use_composition,
    #         description='Use composition if true'
    # )

    launch_args = [
        declare_use_sim_time_cmd,
        declare_use_global_namespace_cmd,
        declare_global_namespace_cmd,
        declare_num_cameras_cmd,
        declare_frame_ids_cmd,
        declare_namespaces_cmd,
        declare_camera_info_files_cmd,
        declare_stream_sources_cmd,
        declare_stream_format_cmd,
        declare_stream_types_cmd,
        declare_fps_cmd,
        declare_widths_cmd,
        declare_heights_cmd,
        declare_gscam_config_cmd,
        declare_save_videos_cmd,
        declare_video_save_filenames_cmd,
        declare_video_save_source_cmd,
        declare_video_save_package_cmd,
        declare_loop_cmd,
        declare_sync_sink_cmd,
        declare_use_gst_timestamps_cmd,
        declare_use_sensor_data_qos_cmd,
        declare_image_encoding_cmd,
        declare_stream_package_cmd,
        declare_model_path_cmd,
        # declare_use_composition_cmd,
    ]

    # Launch nodes
    use_global_namespace_str = use_global_namespace.perform(context)
    global_namespace_str = global_namespace.perform(context)
    num_cameras_int = int(num_cameras.perform(context))
    frame_ids_list =  parse_list_string(frame_ids.perform(context))
    namespaces_list =  parse_list_string(namespaces.perform(context))
    camera_info_files_list = parse_list_string(camera_info_files.perform(context))
    stream_sources_list = parse_list_string(stream_sources.perform(context))
    stream_format_list = parse_list_string(stream_format.perform(context))
    stream_types_list = parse_list_string(stream_types.perform(context))
    fps_list = post_process_list_string(parse_list_string(fps.perform(context)), list_length=num_cameras_int)
    widths_list = post_process_list_string(parse_list_string(widths.perform(context)), list_length=num_cameras_int)
    heights_list = post_process_list_string(parse_list_string(heights.perform(context)), list_length=num_cameras_int)
    gscam_config_list = parse_list_string(gscam_config.perform(context))
    save_videos_str = save_videos.perform(context)
    video_save_filenames_list = parse_list_string(video_save_filenames.perform(context))
    video_save_source_str = str(video_save_source.perform(context))
    video_save_package_str = video_save_package.perform(context)
    loop_str = loop.perform(context)
    sync_sink_str = sync_sink.perform(context)
    use_gst_timestamps_str = use_gst_timestamps.perform(context)
    use_sensor_data_qos_str = use_sensor_data_qos.perform(context)
    image_encoding_str = image_encoding.perform(context)
    stream_package_str = stream_package.perform(context)

    # Ensure lists have the correct length
    assert len(frame_ids_list) == num_cameras_int, "frame_ids list length must match num_cameras"
    assert len(namespaces_list) == num_cameras_int, "namespaces list length must match num_cameras"
    assert len(camera_info_files_list) == num_cameras_int, "camera_info_files list length must match num_cameras"
    assert len(stream_sources_list) == num_cameras_int, "stream_sources list length must match num_cameras"
    assert len(stream_format_list) == num_cameras_int, "stream_format list length must match num_cameras"
    assert len(stream_types_list) == num_cameras_int, "stream_types list length must match num_cameras"
    assert len(gscam_config_list) == num_cameras_int, "gscam_config list length must match num_cameras"
    if save_videos_str.lower() == 'true':
        assert len(video_save_filenames_list) == num_cameras_int, "video_save_filenames list length must match num_cameras"
    assert video_save_source_str in ['raw', 'detection', 'both']
    assert video_save_package_str in ['custom', 'jetson_inference']

    # To handle different detection models
    model_path_str = LaunchConfiguration('model_path', default=os.path.join(image_detection_dir, 'yolo11x.engine')).perform(context)
    model_paths = [model_path_str] * num_cameras_int

    # Generate gscam nodes
    nodes_to_launch = []

    multi_stream_remappings = []
    for i in range(num_cameras_int):
        # Handle launch arguments
        prepend_global_namespace = ''
        if use_global_namespace_str.lower() == "true":
            prepend_global_namespace = global_namespace_str.rstrip('/') + '/'

        # If the config is empty, use stream sources to choose
        if gscam_config_list[i] == "":
            input_codec = 'video/x-raw,'
            if stream_format_list[i] in ("", "raw", "yuyv"):
                input_codec = 'video/x-raw'

            elif stream_format_list[i].lower() in ("jpeg", "mjpg", "mjpeg"):
                # try to detect if the device supports MJPG
                try:
                    # Query the device's supported formats.
                    output = subprocess.check_output(
                            ["v4l2-ctl", "--list-formats-ext", "-d", stream_sources_list[i]],
                            timeout=5).decode("utf-8")
                    # If MJPG is found, use the MJPG pipeline.
                    if "MJPG" in output or "Motion-JPEG" in output:
                        input_codec = 'image/jpeg'
                    else:
                        # If MJPG is not found, use the raw pipeline.
                        input_codec = 'video/x-raw'
                        print(f"Device {stream_sources_list[i]} does not support MJPG. Using raw pipeline.")
                        nodes_to_launch.append(
                                LogInfo(
                                        msg=f"Device {stream_sources_list[i]} does not support MJPG. "
                                            f"Using raw pipeline."))
                except (subprocess.SubprocessError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    # Log or print the error if needed and keep the default raw pipeline.
                    input_codec = 'video/x-raw'
                    nodes_to_launch.append(
                            LogInfo(
                                    msg=f'Could not determine pixel formats for {stream_sources_list[i]}: {e}. '
                                        f'Using raw pipeline.'))

            fps_string = ''
            width_string = ''
            height_string = ''
            if (isinstance(fps_list[i], (float, int)) and fps_list[i] > 0) or (
                    isinstance(fps_list[i], str) and fps_list[i] != "0"):
                fps_string = f',framerate={int(fps_list[i])}/1'
            if (isinstance(widths_list[i], (float, int)) and widths_list[i] > 0) or (
                    isinstance(widths_list[i], str) and widths_list[i] != "0"):
                width_string = f',width={int(widths_list[i])}'
            if (isinstance(heights_list[i], (float, int)) and heights_list[i] > 0) or (
                    isinstance(heights_list[i], str) and heights_list[i] != "0"):
                height_string = f',height={int(heights_list[i])}'

            video_properties = f'{fps_string}{width_string}{height_string}'
            video_convert_str = ' ! videoconvert'
            timestamp_config_string = ''
            output_format_type = image_encoding_to_gstreamer_format.get(image_encoding_str.lower(), 'RGB8')
            output_format_str = f' ! video/x-raw,format={output_format_type}'

            if use_gst_timestamps_str.lower() == "true":
                timestamp_config_string = 'do-timestamp=true'
            jpeg_config = ''
            mjpg_config = ' ! jpegparse ! jpegdec' if stream_format_list[i] in ("jpeg", "mjpg", "mjpeg") else ''
            if image_encoding_str == "jpeg":
                # this block and image_encoding_str == "jpeg" is only used here to match the behaviour one of the GSCAM examples. Use the default image_encoding_str == "rgb8" for normal operation with streams_format="MJPG"
                jpeg_config = ' ! jpegenc ! multipartmux ! multipartdemux ! jpegparse'
                video_convert_str = '' if stream_types_list[i] == "camera" else ' ! videoconvert' # only if the stream_source is a camera
                mjpg_config = ''
                input_codec = 'video/x-raw' if stream_types_list[i] == "camera" else ' ! videoconvert'
                output_format_str = ''

            if stream_types_list[i] == "file":
                gscam_config_list[i] = (f"filesrc location={stream_sources_list[i]} ! decodebin{video_convert_str} ! "
                                        f"videoscale ! videorate ! {input_codec}{video_properties} "
                                        f"{jpeg_config}{mjpg_config}{video_convert_str}{output_format_str}"
                                        )
            elif stream_types_list[i] == "rtsp":
                gscam_config_list[i] = (f"rtspsrc location={stream_sources_list[i]} do-retransmission=false latency=100 buffer-mode=auto ! decodebin{video_convert_str} ! "
                                        f"videoscale ! {input_codec}{width_string}{height_string} "
                                        f"{jpeg_config}{mjpg_config}{video_convert_str}{output_format_str}")
            elif stream_types_list[i] == "camera":
                gscam_config_list[i] = (f"v4l2src {timestamp_config_string} device={stream_sources_list[i]} ! "
                                        f"{input_codec}{video_properties} "
                                        f"{jpeg_config}{mjpg_config}{video_convert_str}{output_format_str}")

        # if save_videos_str.lower() == "true":
        #     # since the GSCam package does not support output sinks, we will use my package ros_images_to_files
        #     mux_type = mux_types.get(video_save_filenames_list[i][-4:], 'matroskamux')
        #     gscam_config_list[i] += (f' ! x264enc ! {mux_type} ! filesink location={video_save_filenames_list[i]} '
        #                              f'async=false')

        # Cleanup the pipeline
        gscam_config_list[i] = clean_gstreamer_pipeline(gscam_config_list[i])

        # Generate gscam node
        gscam_parameters = {
            'frame_id': frame_ids_list[i],
            'camera_info_url': 'file://' + camera_info_files_list[i].strip(),
            'use_sim_time': use_sim_time,
            'gscam_config': gscam_config_list[i],
            'reopen_on_eof': loop,
            'sync_sink': sync_sink,
            'use_gst_timestamps': use_gst_timestamps,
            'use_sensor_data_qos': use_sensor_data_qos,
            'image_encoding': image_encoding
        }

        gscam_node = Node(
                condition=IfCondition(PythonExpression([
                    "'", stream_package, "' == 'gscam'"
                ])),
                package='gscam',
                executable='gscam_node',
                name='gscam_' + str(i),
                namespace=prepend_global_namespace + namespaces_list[i].strip().lstrip('/'),
                output='screen',
                parameters=[
                    gscam_parameters,
                ],
                remappings=[
                    ("camera/image_raw", f"image_raw"),
                    ("camera/camera_info", f"camera_info"),
                    ("camera/image_raw/compressed", "image_raw/compressed"),
                    ("camera/image_raw/compressedDepth", "image_raw/compressedDepth"),
                    ("camera/image_raw/theora", "camera/image_raw/theora"),
                ],
                respawn=True,
                respawn_delay=2.0,
        )
        nodes_to_launch.append(gscam_node)

        # Setup launching streams using jetson_inference package
        if stream_package_str.lower() == 'jetson_inference':
            try:
                # search for the package and catch the exception if it does not exist
                ros_deep_learning_package_share_dir = get_package_share_directory('ros_deep_learning')

                input_resource = stream_sources_list[i]
                if stream_types_list[i] == "file":
                    if not stream_sources_list[i].startswith("file://"):
                        input_resource = f"file://{stream_sources_list[i]}"
                elif stream_types_list[i] == "rtsp":
                    if not stream_sources_list[i].startswith("rtsp://"):
                        input_resource = f"rtsp://{stream_sources_list[i]}"
                elif stream_types_list[i] == "camera":
                    if not stream_sources_list[i].startswith("v4l2://"):
                        input_resource = f"v4l2://{stream_sources_list[i]}"  # v4l2 (optional), csi

                jetson_inference_node = Node(
                    # condition=IfCondition(PythonExpression([
                    #     "'", stream_package, "' == 'jetson_inference'"
                    # ])),
                    package='ros_deep_learning',
                    executable='video_source',
                    name='jetson_inference_' + str(i),
                    namespace=prepend_global_namespace + namespaces_list[i].strip().lstrip('/'),
                    output='screen',
                    parameters=[
                        {
                            'use_sim_time': use_sim_time,
                            'resource': input_resource,
                            'width': int(widths_list[i]),
                            'height': int(heights_list[i]),
                            'codec': "unknown",
                            'loop': -1 if loop_str.lower() == 'true' else 0,
                            'latency': 0,  # 2000
                            'framerate': float(fps_list[i])
                        }
                    ],
                    remappings=[
                        ("raw", f"image_raw"),
                    ],
                    respawn=True,
                    respawn_delay=2.0,
                )
                nodes_to_launch.append(jetson_inference_node)

            except ROS2PackageNotFoundError as e:
                error_msg = LogInfo(msg=f'Failed to launch ros_deep_learning: {e}.')
                nodes_to_launch.append(error_msg)

        image_topic_is_compressed = False
        image_topic = 'image_raw'
        if image_encoding_str == "jpeg":
            image_topic_is_compressed = True
            image_topic += '/compressed'

        detection_image_topic = 'yolo/detection_image'
        # Launch video recording via my ros_images_to_files package since gscam does not support output sinks
        if save_videos_str.lower() == "true":
            valid_options = ('_raw', '_detection')
            output_file_name_orig = video_save_filenames_list[i]
            output_file_name_clean = output_file_name_orig.split('://')  # to remove file://, rtsp://, etc
            output_file_path_obj = pathlib.Path(output_file_name_clean[-1])

            video_record_topic = []
            if video_save_source_str.lower() == "raw":
                video_record_topic.append(image_topic)
            elif video_save_source_str.lower() == "detection":
                video_record_topic.append(detection_image_topic)
            elif video_save_source_str.lower() == "both":
                video_record_topic.extend([image_topic, detection_image_topic])

            for vid_record_idx, vid_record_topic in enumerate(video_record_topic):
                # modify the file name with more info
                str_to_append = ''
                if video_save_source_str.lower() == "raw":
                    # str_to_append = valid_options[0] if len(video_record_topic) > 1 else ''
                    pass
                elif video_save_source_str.lower() == "detection":
                    # str_to_append = valid_options[1] if len(video_record_topic) > 1 else ''
                    pass
                elif video_save_source_str.lower() == "both":
                    str_to_append = valid_options[vid_record_idx]
                output_file_name = str(output_file_path_obj.with_stem(f"{output_file_path_obj.stem}{str_to_append}"))
                output_file_name = output_file_name if len(output_file_name_clean) == 1 else f'{output_file_name_clean[0]}://{output_file_name}'

                if video_save_package_str.lower() == 'custom':
                    try:
                        # search for the package and catch the exception if it does not exist
                        ros_images_to_files_package_share_dir = get_package_share_directory('ros_images_to_files')
                        custom_video_recorder_node = Node(
                                package='ros_images_to_files',
                                executable='video_recorder_node',
                                name=f'custom_video_recorder_cam{i}{valid_options[vid_record_idx]}',
                                namespace=prepend_global_namespace + namespaces_list[i].strip().lstrip('/'),
                                output='screen',
                                parameters=[
                                    {'use_sim_time': use_sim_time},
                                    {'image_topic': vid_record_topic},
                                    {'image_topic_is_compressed': image_topic_is_compressed},
                                    {'output_file_name': output_file_name.split('://')[-1]},
                                    {'queue_size': 100},
                                    {'fps': float(fps_list[i]) if fps_list[i] > 0 else 30.0},
                                    {'qos': 'SENSOR_DATA' if use_sensor_data_qos_str.lower() == "true" else 'SYSTEM_DEFAULT'},
                                    {'show_image': False},
                                ]
                        )
                        nodes_to_launch.append(custom_video_recorder_node)
                    except ROS2PackageNotFoundError as e:
                        error_msg = LogInfo(msg=f'Failed to launch custom_video_recorder_node: {e}. Skipping video recording.')
                        nodes_to_launch.append(error_msg)

                elif video_save_package_str.lower() == 'jetson_inference':
                    try:
                        # search for the package and catch the exception if it does not exist
                        ros_deep_learning_package_share_dir = get_package_share_directory('ros_deep_learning')
                        # if not output_file_name.startswith(("file://", "rtsp://")):
                        #     outpuoutput_file_namet_resource = f"file://{output_file_name[i]}"

                        jetson_inference_video_recorder_node = Node(
                            package='ros_deep_learning',
                            executable='video_output',
                            name=f'jetson_inference_video_recorder_cam{i}{valid_options[vid_record_idx]}',
                            namespace=prepend_global_namespace + namespaces_list[i].strip().lstrip('/'),
                            output='screen',
                            parameters=[
                                {
                                    'use_sim_time': use_sim_time,
                                    'resource': output_file_name,
                                    'codec': 'unknown',
                                    'bitrate': 0
                                },

                            ],
                            remappings=[
                                ("image_in", vid_record_topic),
                            ],
                        )
                        nodes_to_launch.append(jetson_inference_video_recorder_node)
                    except ROS2PackageNotFoundError as e:
                        error_msg = LogInfo(msg=f'Failed to launch ros_deep_learning: {e}.')
                        nodes_to_launch.append(error_msg)

        # Generate autodriver_image_object_detection nodes
        yolo_node = Node(
            package= 'autodriver_image_object_detection',
            executable= 'single_stream_detector',  # yolo_detector
            name= 'yolo_detection_node_' + str(i),
            namespace= prepend_global_namespace + namespaces_list[i].strip().lstrip('/'),
            output= 'screen',
            parameters= [
                {
                    'use_sim_time': use_sim_time,
                    'input_image_topic': image_topic,
                    'input_camera_info_topic': 'camera/camera_info',
                    'input_image_topic_is_compressed': image_topic_is_compressed,
                    'detection_results_topic': 'yolo/detection_results',
                    'publish_debug_image': True,
                    'detection_image_topic': detection_image_topic,
                    'segmentation_image_topic': 'yolo/segmentation_image',
                    'segmentation_mask_image_topic': 'yolo/segmentation_mask_image',
                    'qos': 'SENSOR_DATA' if use_sensor_data_qos_str.lower() == "true" else 'SYSTEM_DEFAULT',
                    # yolo11x.engine, yolo12x.engine, rtdetr-x.pt, yolov9e.engine, yolo11m-seg.engine, yolov10x.engine, yolov8x.engine
                    'model_path': model_paths[i],
                    'export_model_format': '',
                    'use_image_dimensions': False,  # True (slower but works with dynamic exports), False (faster with fixed size/batch exports)
                    'image_dimensions': [640, 640],  # [480, 640]
                    'resize_image': False,  # False
                    'use_gpu': True,
                    'show_image': False,
                    'conf_thresh': 0.55,  # 0.55
                    'iou_thresh': 0.55,
                    'max_det': 100,
                    'augment': False,
                    'queue_size': 1,
                    'tracker_2d.path': os.path.join(image_detection_config_dir, 'tracker_custom.yaml'),
                    'tracker_2d.tracker_type': 'bytetrack',  # bytetrack
                    'tracker_2d.track_high_thresh': 0.45,  # -1.0
                    'tracker_2d.track_low_thresh': -1.0,
                    'tracker_2d.new_track_thresh': -0.5,  # -1.0
                    'tracker_2d.track_buffer': -1,
                    'tracker_2d.match_thresh': 0.95,  # -1.0
                    'tracker_2d.fuse_score': True,
                    'tracker_2d.gmc_method': '',
                    'tracker_2d.proximity_thresh': -1.0,
                    'tracker_2d.appearance_thresh': -1.0,
                    'tracker_2d.with_reid': False,  # False
                    'tracker_2d.model': 'auto',
                }
            ]
        )
        nodes_to_launch.append(yolo_node)

        multi_stream_remappings.append((f'stream_{i}/image_raw', f"{namespaces_list[i].strip().lstrip('/')}/image_raw"))
        multi_stream_remappings.append((f'stream_{i}/camera_info', f"{namespaces_list[i].strip().lstrip('/')}/camera_info"))

        # add tracking node
        tracking_node = Node(
                package='autodriver_image_object_detection',
                executable='tracking_node_2d',
                name='tracking_node_' + str(i),
                namespace=prepend_global_namespace,
                output='screen',
                parameters=[
                    {
                        'use_sim_time': use_sim_time,
                    }
                ],
                remappings=[
                    ("image_raw", f"{namespaces_list[i].strip().lstrip('/')}/image_raw"),
                    ("detections_2d", f"stream_{i}/yolo/detection/results"),
                    ("tracked_detections_2d", "tracked_detections_2d_" + str(i))
                ]
        )
        #nodes_to_launch.append(tracking_node)

    # multi stream detector
    multi_yolo_node = Node(
            package='autodriver_image_object_detection',
            executable='multi_stream_detector',  # yolo_detector
            name='multi_yolo_detection_node',
            namespace=prepend_global_namespace,
            output='screen',
            parameters=[
                {
                    'use_sim_time': use_sim_time,
                    'num_cameras': num_cameras,
                    'synchronization_interval': 0.1,
                    'input_image_topic_is_compressed': [False] * num_cameras_int,
                    'qos': 'SENSOR_DATA' if use_sensor_data_qos_str.lower() == "true" else 'SYSTEM_DEFAULT',
                    'model_path': "yolo11m-seg.engine",  # rtdetr-l.pt, yolo11m-seg.engine
                    'export_model_format': '',
                    'use_gpu': True,
                    'show_image': False,
                }
            ],
            remappings=multi_stream_remappings
    )
    #nodes_to_launch.append(multi_yolo_node)

    # RViz node
    rviz_node = Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                # arguments=['-d', 'detection.rviz'],
                output='screen'
            )
    nodes_to_launch.append(rviz_node)

    # return the launch description
    camera_group = GroupAction(
            actions=[
                SetParameter(name='use_sim_time', value=use_sim_time),
                # nodes
                *nodes_to_launch
            ]
    )
    ld = launch_args + [camera_group]
    return ld


def generate_launch_description():
    return LaunchDescription(
            [
                SetEnvironmentVariable(name='RCUTILS_COLORIZED_OUTPUT', value='1'),
                OpaqueFunction(function=launch_setup)
            ]
    )
