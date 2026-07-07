"""
ROS2 node for obstacle detection using Euclidean Clustering on Point Clouds.
Publishes clustered pointcloud, visualization markers, and object array.

Usage:
    * sudo apt-get install ros-${ROS_DISTRO}-derived-object-msgs ros-${ROS_DISTRO}-vision-msgs
    * (optional) export LD_PRELOAD=${HOME}/sdks/open3d_install/lib/libOpen3D.so # to fix open3d python ImportError
    * ros2 run autodriver_image_object_detection yolo_detection_node

1. Subscribe to image, pointcloud, depth using message filters
2. Detect obstacles using YOLO [done]
3. Publish detections to vision_msgs/Detection2DArray and Image [done]
4. Implement projection to 3D [done]

Todo:
    * refactor this node so it can work with/without depth/pointcloud, i.e cleanly separate image stuff, depth and pointcloud
    * Implement non-tracking (i.e predict) [done]
    * Setup segmentation (https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Masks | )
        * show segmentation mask image [done]
        * publish segmentation mask image [done]
        * resize the mask to the original shape (or depth image shape) [done: no need]
    * Setup 3D [depth] [done]
    * transform the depth masks/bboxes to the base_link frame [done]
    * Setup 3D [done]
    * transform the pointcloud masks/bboxes to the base_link frame [done]
    * add pre transformed points to the pointcloud to avoid multiple transformations and see if it improves performance [done: it does]
    * add check for .engine model with try-catch and then convert to tensorrt [done]
    * switch to engine models as the default [done]
    * add ability to run multiple models simultaneously, e.g segmentation and obb [done: just run another instance of this node for separation of concern reasons]
    * use TimeSynchronizer if synchronization_interval == 0.0 or if approx_sync parameter is False [done]
    * setup limiting detected classes [done]
    * plot 2D tracks over time [done]
    * export models to engine
    * publish the axes (/tf of detected objects)

    * add a parameter to choose what timestamp should be put in the message (current time or message timestamp)
    * Publish clusters/pointclouds
    * Fix projection accuracy and compare, e.g with Carla
        * use the following example for 2D detection projection to 3D (https://github.com/tony23545/nav2_dynamic_obstacle/blob/master/detectron2_detector/detectron2_detector/detectron2_node.py)
        * copy my new torch reprojections
        * Ask GPTs
    * Setup OBB (https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.OBB)
    * add support for batch inference (multiple cameras/images at once)
    * filter out objects/clusters with min_height above a certain threshold
    * publish as a derived_object

    * Cleanup
    * rename rgb to camera_0
    * rename images to frame (to accomodate pointcloud)
    * publish detection pointcloud for debugging
    * switch to image transport for more modularity and better compressed image support use try-catch to handle image transport not being installed (https://github.com/ros-perception/image_transport_tutorials?tab=readme-ov-file#py_simple_image_pub)

Bugs:
    * Depth:
        * 3D bounding box from depth too long, probable caused by far away objects intersecting with the 2D bounding box
        * 3D bounding box dimensions not right. Needs fixing
"""
import os
import sys
import time
import struct
import tempfile
from collections import defaultdict

import yaml
import numpy as np
try:
    import scipy
    from scipy.spatial.transform import Rotation as R
    SCIPY_INSTALLED = True
    SCIPY_VERSION = scipy.__version__
except ImportError:
    SCIPY_INSTALLED = False
    SCIPY_VERSION = '0.0.0'

try:
    import tf_transformations
    from tf_transformations import quaternion_matrix, quaternion_from_matrix
    TF_TRANSFORMATIONS_INSTALLED = True
except ImportError:
    TF_TRANSFORMATIONS_INSTALLED = False

import cv2
import torch
import torch.utils.dlpack
import ultralytics

try:
    import open3d as o3d
    import open3d.core as o3c
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from rclpy import qos
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from std_msgs.msg import Header, ColorRGBA
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, Imu, PointCloud2, PointField
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2
from sensor_msgs_py.point_cloud2 import read_points, create_cloud
from image_geometry import PinholeCameraModel
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
from derived_object_msgs.msg import Object, ObjectArray
from shape_msgs.msg import SolidPrimitive
try:
    from nav2_dynamic_msgs.msg import Obstacle, ObstacleArray
    NAV2_DYNAMIC_MSGS_AVAILABLE = True
except ImportError:
    NAV2_DYNAMIC_MSGS_AVAILABLE = False
import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, LookupException, ConnectivityException, \
    ExtrapolationException

from autodriver_image_object_detection.utils.common import pack_2d_detection, pack_nav2_obstacle_msg, pack_derived_object_msg, update_tracker_param, make_deleteall_marker_array
from autodriver_image_object_detection.utils.imaging_utils import parse_image_message as _parse_image_message
from autodriver_image_object_detection.utils.profiling import setup_profiler, apply_profiler_param

if OPEN3D_AVAILABLE:
    from autodriver_pointcloud_preprocessor.pointcloud_preprocessor import PointcloudPreprocessorNode


    # from autodriver_pointcloud_preprocessor.utils import (convert_pointcloud_to_numpy, numpy_struct_to_pointcloud2,
    #                                                       get_current_time, get_time_difference,
    #                                                       dict_to_open3d_tensor_pointcloud,
    #                                                       pointcloud_to_dict, get_pointcloud_metadata,
    #                                                       check_field, crop_pointcloud,
    #                                                       extract_rgb_from_pointcloud, get_fields_from_dicts,
    #                                                       remove_duplicates, rgb_float_to_bytes,
    #                                                       FIELD_DTYPE_MAP, FIELD_DTYPE_MAP_INV)


class ImageObstacleDetectionNode(Node):
    def __init__(self):
        this_package_dir = get_package_share_directory('autodriver_image_object_detection')
        super(ImageObstacleDetectionNode, self).__init__("image_obstacle_detection_node")

        # Declare parameters
        self.declare_parameter(name='input_image_topic', value="carla/ego_vehicle/rgb_front/image",  # "camera/image_raw", camera/color/image_raw, carla/ego_vehicle/rgb_front/image
                               descriptor=ParameterDescriptor(
                                   description='The input image topic. '
                                               'Works with all image types: RGB(A), BGR(A), mono8, mono16.',
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='input_camera_info_topic', value="carla/ego_vehicle/rgb_front/camera_info",  # camera/color/camera_info, carla/ego_vehicle/rgb_front/camera_info
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='input_image_topic_is_compressed', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='detection_results_topic', value="yolo/detection_results",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter(name='detection_image_topic', value="yolo/detection_image",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='segmentation_image_topic', value="yolo/segmentation_image",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='segmentation_mask_image_topic', value="yolo/segmentation_mask_image",
                               descriptor=ParameterDescriptor(
                                   description='',
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='depth_image_topic', value="carla/ego_vehicle/depth_front/image",
                               # "depth/image_raw", camera/aligned_depth_to_color/image_raw, carla/ego_vehicle/depth_front/image
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='depth_camera_info_topic',
                               value="carla/ego_vehicle/depth_front/camera_info",  # "depth/camera_info", carla/ego_vehicle/depth_front/camera_info
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='pointcloud_topic',
                               value="carla/ego_vehicle/lidar",  # camera/depth/color/points, carla/ego_vehicle/lidar
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
            description='',
            type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='model_path', value="yolo11x-seg.pt",  # yolo11n-seg.pt, yolo11x-seg.pt, yolo11x.pt
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='export_model_format', value='',
                               descriptor=ParameterDescriptor(
                                       description='Export the model to one of the supported formats '
                                                   'if the file does not exist. '
                                                   'See https://docs.ultralytics.com/modes/export/#export-formats '
                                                   'for supported formats.',
                               type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('track_2d', True)
        self.declare_parameter('tracker_2d.path', os.path.join(this_package_dir, 'config', 'tracker_custom.yaml'))
        self.declare_parameter('tracker_2d.tracker_type', 'bytetrack')
        self.declare_parameter('tracker_2d.track_high_thresh', -1.0)
        self.declare_parameter('tracker_2d.track_low_thresh', -1.0)
        self.declare_parameter('tracker_2d.new_track_thresh', -1.0)
        self.declare_parameter('tracker_2d.track_buffer', -1)
        self.declare_parameter('tracker_2d.match_thresh', -1.0)
        self.declare_parameter('tracker_2d.fuse_score', True)
        self.declare_parameter('tracker_2d.gmc_method', '')
        self.declare_parameter('tracker_2d.proximity_thresh', -1.0)
        self.declare_parameter('tracker_2d.appearance_thresh', -1.0)
        self.declare_parameter('tracker_2d.with_reid', True)
        self.declare_parameter('tracker_2d.model', 'auto')
        self.declare_parameter('plot_tracks', True)
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('track_3d', True)  # todo: implement 3D tracking
        self.declare_parameter('synchronization_interval', 0.1)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter(name='show_image', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name="use_image_dimensions", value=True, descriptor=ParameterDescriptor(
            description='Whether to use the image dimensions when running inference or using a fixed square image '
                        'size for the model. '
                        'Setting to True typically yields better performance when running models exported with '
                        'dynamic shapes but is generally slower with False. '
                        'False: faster with fixed size/batch exports.',
            type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name="image_dimensions", value=[640, 640], descriptor=ParameterDescriptor(
            description='The image dimensions to use when running inference. '
                        'Must be set if exporting the model to another format, '
                        'e.g TensorRT .engine since that is compiled with a fixed size. '
                        'Examples: [480, 640], [1080, 1920]. Default: [640, 640].',
            type=ParameterType.PARAMETER_INTEGER_ARRAY
        ))
        self.declare_parameter("resize_image", False)
        self.declare_parameter("half_precision", True)
        self.declare_parameter("conf_thresh", 0.55)
        self.declare_parameter("iou_thresh", 0.55)
        self.declare_parameter("max_det", 50)
        self.declare_parameter("classes", ['person', 'car', 'chair'])  # [] or ['person', 'car'] or [0, 2]
        self.declare_parameter("update_class", "")  # type "class_name" to add or "-class_name" to delete
        self.declare_parameter("agnostic_nms", True)
        self.declare_parameter("augment", False)
        self.declare_parameter("verbose", False)
        self.declare_parameter('static_camera_info', True)
        self.declare_parameter("transform_timeout", 2.0)  # 0.1
        self.declare_parameter("project_to_3d", True)  # todo: remove this flag and just use depth or pointcloud
        self.declare_parameter("publish_empty_detections", True)  # heartbeat: publish empty 3D + DELETEALL markers on zero-detection frames
        self.declare_parameter("use_depth",
                               True)  # can run both at the same time at the cost of speed. Depth is significantly faster for now (about 2.5 times)
        self.declare_parameter("use_pointcloud",
                               True)  # can run both at the same time at the cost of speed. Depth is significantly faster for now (about 2.5 times)
        self.declare_parameter("output_frame", "ego_vehicle")  # e.g base_link, ego_vehicle (carla)
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter('depth_scale', 1.0)  # mm to meters. 1.0 for Carla (32FC1), 1000.0 for Realsense (16UC1)
        self.declare_parameter('depth_max', 50.0)  # meters. 1000. for Carla, 6.0 for Realsense
        # upper bound on the 3D box extent along the camera view axis (meters);
        # rejects background depth pixels bleeding through the mask edges
        self.declare_parameter('depth_box_thickness', 4.0)
        self.declare_parameter(name='normalize_depth', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='normalized_max', value=255, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_DOUBLE))

        self.declare_parameter('crop_to_roi', False)
        self.declare_parameter('roi_min', [-60.0, -60.0, -20.0])  # [-6.0, -6.0, 0.0]
        self.declare_parameter('roi_max', [60.0, 60.0, 20.0])  # [6.0, 6.0, 2.0]
        self.declare_parameter('voxel_size', 0.05)  # 0.01, 0.05
        self.declare_parameter('remove_statistical_outliers', False)
        self.declare_parameter('estimate_normals', False)
        self.declare_parameter('remove_ground', False)
        self.declare_parameter("cluster_tolerance", 1.0)  # meters. 0.2. Carla 1.0
        self.declare_parameter("min_cluster_size", 5)  # 100. Carla (5)
        self.declare_parameter("max_cluster_size", 1000)  # values <= 0 means no limit
        self.declare_parameter('cluster_min_height', 0.1)  # min height of cluster
        self.declare_parameter('cluster_max_height', 2.0)  # max height of cluster
        self.declare_parameter("bounding_box_type", "AABB")  # AABB or OBB
        self.declare_parameter('publish_object_array', True)
        self.declare_parameter('publish_obstacle_array', True)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.publish_object_array = self.get_parameter('publish_object_array').get_parameter_value().bool_value
        self.publish_obstacle_array = self.get_parameter('publish_obstacle_array').get_parameter_value().bool_value
        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.input_camera_info_topic = self.get_parameter('input_camera_info_topic').value
        self.input_image_topic_is_compressed = self.get_parameter('input_image_topic_is_compressed').value
        self.detection_results_topic = self.get_parameter('detection_results_topic').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.detection_image_topic = self.get_parameter('detection_image_topic').value
        self.segmentation_image_topic = self.get_parameter('segmentation_image_topic').value
        self.segmentation_mask_image_topic = self.get_parameter('segmentation_mask_image_topic').value
        self.depth_image_topic = self.get_parameter('depth_image_topic').value
        self.depth_camera_info_topic = self.get_parameter('depth_camera_info_topic').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.qos = self.get_parameter('qos').value
        self.model_path = self.get_parameter('model_path').value
        self.export_model_format = self.get_parameter('export_model_format').get_parameter_value().string_value
        self.track_2d = self.get_parameter('track_2d').get_parameter_value().bool_value
        self.tracker_2d_cfg = self.get_parameters_by_prefix('tracker_2d')
        self.tracker_2d_cfg = {k: v.value for k, v in self.tracker_2d_cfg.items()}
        self.plot_tracks = self.get_parameter('plot_tracks').value
        self.queue_size = self.get_parameter('queue_size').value
        self.track_3d = self.get_parameter('track_3d').value
        self.synchronization_interval = self.get_parameter('synchronization_interval').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.use_image_dimensions = self.get_parameter("use_image_dimensions").get_parameter_value().bool_value
        self.image_dimensions = self.get_parameter("image_dimensions").get_parameter_value().integer_array_value
        self.resize_image = self.get_parameter("resize_image").get_parameter_value().bool_value
        self.half_precision = self.get_parameter("half_precision").get_parameter_value().bool_value
        self.conf_thresh = self.get_parameter("conf_thresh").get_parameter_value().double_value
        self.iou_thresh = self.get_parameter("iou_thresh").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.classes = (
            self.get_parameter("classes").value
        )
        self.update_class = self.get_parameter("update_class").value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.verbose = self.get_parameter("verbose").get_parameter_value().bool_value
        self.project_to_3d = self.get_parameter("project_to_3d").get_parameter_value().bool_value
        self.publish_empty_detections = self.get_parameter("publish_empty_detections").get_parameter_value().bool_value
        self.use_depth = self.get_parameter("use_depth").get_parameter_value().bool_value
        self.use_pointcloud = self.get_parameter("use_pointcloud").get_parameter_value().bool_value
        self.output_frame = self.get_parameter("output_frame").value
        self.static_camera_to_robot_tf = self.get_parameter(
            "static_camera_to_robot_tf").get_parameter_value().bool_value
        self.transform_timeout = self.get_parameter("transform_timeout").get_parameter_value().double_value
        self.static_camera_info = self.get_parameter("static_camera_info").get_parameter_value().bool_value
        self.depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        self.depth_max = self.get_parameter("depth_max").get_parameter_value().double_value
        self.depth_box_thickness = self.get_parameter(
            "depth_box_thickness").get_parameter_value().double_value
        self.normalize_depth = self.get_parameter("normalize_depth").get_parameter_value().bool_value
        self.normalized_max = self.get_parameter("normalized_max").get_parameter_value().double_value

        # todo: use pointcloud preprocessing class
        self.crop_to_roi = self.get_parameter('crop_to_roi').value
        self.roi_min = self.get_parameter('roi_min').value
        self.roi_max = self.get_parameter('roi_max').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.remove_statistical_outliers = self.get_parameter('remove_statistical_outliers').value
        self.estimate_normals = self.get_parameter('estimate_normals').value
        self.remove_ground = self.get_parameter('remove_ground').value
        self.cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        self.max_cluster_size = self.get_parameter("max_cluster_size").get_parameter_value().integer_value
        self.cluster_min_height = self.get_parameter('cluster_min_height').value
        self.cluster_max_height = self.get_parameter('cluster_max_height').value
        self.bounding_box_type = self.get_parameter("bounding_box_type").value

        os.environ['YOLO_VERBOSE'] = str(self.verbose)

        if not OPEN3D_AVAILABLE:
            self.get_logger().warn("Open3D not installed. PointCloud use is disabled.")
            self.use_pointcloud = False

        # Setup the device
        self.device = 'cpu'
        self.torch_device = torch.device('cpu')
        if self.project_to_3d and self.use_pointcloud:
            self.o3d_device = o3d.core.Device('CPU:0')

        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.torch_device = torch.device('cuda:0')
            if self.project_to_3d and self.use_pointcloud:
                # todo: add separate flag for Open3D
                if o3d.core.cuda.is_available():
                    self.o3d_device = o3d.core.Device('CUDA:0')

        # Initialize variables
        self.image_frame_id = None
        self.image_width = None
        self.image_height = None
        self.imgsz = None
        self.bridge = CvBridge()
        self.use_segmentation = "seg" in self.model_path
        self.task = "segment" if self.use_segmentation else "detect"

        if self.plot_tracks:
            # Store the track history
            self.track_history = defaultdict(lambda: [])

        model_architectures = {
            'yolo': ultralytics.YOLO,  # yolov8n-seg.pt, yolo11n-seg.pt, YOLO12n-seg.pt, yoloe-s.pt
            'rtdetr': ultralytics.RTDETR,  # rtdetr-l.pt
            'nas': ultralytics.NAS,  # yolo_nas_s.pt
            'worldv2': ultralytics.YOLOWorld,  # yolov8s-worldv2.pt
        }

        if 'rtdetr' in self.model_path:
            model_class = model_architectures['rtdetr']
        elif 'nas' in self.model_path:
            model_class = model_architectures['nas']
        elif 'worldv2' in self.model_path:
            model_class = model_architectures['worldv2']
        else:
            model_class = model_architectures['yolo']

        imgsz = self.image_dimensions if self.use_image_dimensions else (640, 640)
        # (optional) export the model
        if self.export_model_format:
            self.get_logger().info(f"Exporting model to {self.export_model_format} format...")
            self.model = model_class(
                    self.model_path.split('.')[0] + '.pt',
                    # task=self.task,
            )  # can only export pytorch models
            self.model.export(
                    format=self.export_model_format, half=self.half_precision, simplify=True, nms=self.iou_thresh > 0.0,
                    # imgsz=tuple(imgsz),  # not necessary if dynamic=True
                    dynamic=True,
                    device=self.device
            )

            self.get_logger().info(f"Exported model to {self.export_model_format} format: {self.model_path}")
            self.model_path = self.model_path.rsplit('.', 1)[0] + '.' + self.export_model_format  # self.model_path.rsplit('.', 1)[:-1]

        # if model_path ends with .engine or .onnx, try loading the file and export if FileNotFoundError
        if self.model_path.split('.')[-1] in ['engine', 'onnx']:
            try:
                self.model = model_class(
                        self.model_path,
                        # task=self.task,
                )
            except FileNotFoundError:
                self.get_logger().info(f"Model not found: {self.model_path}. "
                                       f"Trying to export to {self.model_path.split('.')[-1]}.")

                self.model = model_class(
                        self.model_path.split('.')[0] + '.pt',
                        # task=self.task,
                )  # append .pt to the model path
                self.model.export(
                        format=self.model_path.split('.')[-1],
                        half=self.half_precision,
                        simplify=True,
                        nms=self.iou_thresh > 0.0,
                        # imgsz=tuple(imgsz),  # not necessary if dynamic=True
                        dynamic=True,
                        device=self.device
                )
                self.model_path = self.model_path.split('.')[0] + '.' + self.model_path.split('.')[-1]

        # Initialize model
        self.model = model_class(
                self.model_path,
                # task=self.task,
        )
        try:
            self.model.to(self.torch_device)
        except TypeError:
            pass

        # Filter classes
        self.class_names: dict[int, str] = self.model.names
        num_model_classes = len(self.class_names)
        self.class_names_inv = {v: k for k, v in self.class_names.items()}
        self.supported_class_names = set(self.class_names_inv.keys())
        self.supported_class_keys = set(self.class_names.keys())
        if len(self.classes) == 0:
            self.classes = list(range(num_model_classes))
        else:
            if isinstance(self.classes, int):
                assert self.classes < num_model_classes
                self.classes = [self.classes]
            elif isinstance(self.classes, str):
                self.classes = [int(x.strip()) for x in
                                self.classes.split(',')]  # assert all ints less than num_model_classes
                assert all(x < num_model_classes for x in self.classes)
            elif isinstance(self.classes, list):
                # remove empty strings from the list but keep 0
                self.classes = [desired_class for desired_class in self.classes if desired_class or desired_class == 0]
                # if classes is a list of strings
                if isinstance(self.classes[0], str):
                    assert all(x in self.supported_class_names for x in self.classes)
                    self.classes = [self.class_names_inv[x.strip()] for x in self.classes]
                # if classes is a list of ints
                elif isinstance(self.classes[0], int):
                    assert all(x in self.supported_class_keys for x in self.classes)
                else:
                    raise ValueError("Classes must either be a list of ints or a strings.")
            else:
                self.classes = list(self.classes)

        self.get_logger().info(f"Only detecting classes: {[self.class_names[class_] for class_ in self.classes]}")

        self.results = None
        self.detection_image = None
        self.cameras = ['rgb']
        self.camera_infos = {'rgb': None}
        self.camera_models = {'rgb': PinholeCameraModel()}
        self.images = {'rgb': None}  # todo: rename rgb to frame
        self.frame_ids = {'rgb': None}
        self.headers = {'rgb': None}
        self.msg_metadata = {'rgb': None}
        if self.project_to_3d:
            if self.use_depth:
                self.cameras.append('depth')
                self.camera_infos['depth'] = None
                self.camera_models['depth'] = PinholeCameraModel()
                depth_dict = {'depth': None, 'rgbd': None}
                # depth->rgb (for depth/RGB alignment) and depth->output_frame (for
                # lifting optical-frame boxes) caches. Kept separate from the
                # pointcloud path's output_frame_to_rgb_tf* cache — sharing one
                # variable let whichever path ran first poison the other.
                self.depth_to_rgb_tf = None
                self.depth_to_rgb_tf_torch = None
                self.depth_to_output_tf = None
                self.images.update(depth_dict)
                self.frame_ids.update(depth_dict)
                self.headers.update(depth_dict)
                self.msg_metadata.update(depth_dict)
                self.previous_rgbd_image = None

            if self.use_pointcloud:
                self.images['pointcloud'] = None
                self.frame_ids['pointcloud'] = None
                self.headers['pointcloud'] = None
                self.msg_metadata['pointcloud'] = {}
                self.o3d_camera_intrinsics = {'rgb': None,
                                              'depth': None} if self.use_depth else {'rgb': None}
                self.o3d_camera_models = {'rgb': o3d.camera.PinholeCameraIntrinsic(),
                                          'depth': o3d.camera.PinholeCameraIntrinsic()}
                self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)
                self.camera_to_robot_tf_o3d = None
                self.output_frame_to_rgb_tf = None
                self.output_frame_to_rgb_tf_o3d = None
                self.previous_pointcloud = None
                # pointcloud unpacking variables. todo: use my pointcloud preprocessor class to handle this later
                self.pointcloud_preprocessor_namespace = 'detection_pointcloud_preprocessor'
                self.pointcloud_preprocessor = PointcloudPreprocessorNode(
                        node_name=self.pointcloud_preprocessor_namespace, enabled=False,
                        parameter_namespace=self.pointcloud_preprocessor_namespace)
                # to get the dict of all parameters: self.pointcloud_preprocessor.get_parameters_by_prefix(prefix=self.pointcloud_preprocessor_namespace.rstrip('.'))
                self.pointcloud_preprocessor_namespace_param = f'{self.pointcloud_preprocessor_namespace}.'
                preprocessor_params = (
                        [
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}use_gpu', Parameter.Type.BOOL,
                                      self.use_gpu),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}transform_pointcloud',
                                      Parameter.Type.BOOL,
                                      True),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}transform_before_preprocessing',
                                      Parameter.Type.BOOL,
                                      False),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}robot_frame', Parameter.Type.STRING,
                                      self.output_frame),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}static_camera_to_robot_tf', Parameter.Type.BOOL,
                                      self.static_camera_to_robot_tf),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}transform_timeout',
                                      Parameter.Type.DOUBLE,
                                      self.transform_timeout),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}crop_to_roi', Parameter.Type.BOOL, self.crop_to_roi),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}roi_min', Parameter.Type.DOUBLE_ARRAY,
                                      self.roi_min),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}roi_max',
                                      Parameter.Type.DOUBLE_ARRAY,
                                      self.roi_max),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}voxel_size',
                                      Parameter.Type.DOUBLE,
                                      self.voxel_size),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}remove_statistical_outliers', Parameter.Type.BOOL,
                                      self.remove_statistical_outliers),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}estimate_normals',
                                      Parameter.Type.BOOL,
                                      self.estimate_normals),
                            Parameter(f'{self.pointcloud_preprocessor_namespace_param}remove_ground',
                                      Parameter.Type.BOOL,
                                      self.remove_ground),
                        ]
                )
                # The child PointcloudPreprocessorNode may not have declared these namespaced
                # params (e.g. when constructed with enabled=False), so a plain set_parameters()
                # raises ParameterNotDeclaredException. Declare-if-missing then set, so config
                # works regardless of the preprocessor's declaration timing.
                for _pp in preprocessor_params:
                    if self.pointcloud_preprocessor.has_parameter(_pp.name):
                        self.pointcloud_preprocessor.set_parameters([_pp])
                    else:
                        self.pointcloud_preprocessor.declare_parameter(_pp.name, _pp.value)

        self.previous_time = time.time()
        self.previous_callback_time = None
        self.last_timestamp = None
        self.camera_to_robot_tf = None

        try:
            self.get_logger().info("Fusing model...")
            self.model.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fusing the model: {e}. "
                                   f"This usually occurs if not using a pytorch model (.pt), "
                                   f"e.g a TensorRT model (.engine)")

        # A fused .pt model moved to GPU keeps fp32 weights, while half=True feeds fp16
        # inputs — this mismatch causes a "Half != float" runtime error. Cast weights to
        # fp16 so they match. Engines are already fp16 so this path is .pt-only.
        if (self.torch_device.type != 'cpu' and self.half_precision
                and self.model_path.endswith('.pt')):
            try:
                self.model.model.half()
            except Exception as e:
                self.get_logger().warn(
                    f'Could not cast .pt model to fp16 ({e}); falling back to fp32 inference.')
                self.inference_dict['half'] = False

        # Initialize TF buffer and listener. spin_thread=True so /tf keeps flowing
        # while a blocking lookup_transform waits inside the detection callback;
        # otherwise the single-threaded executor starves the TF subscription and
        # every lookup stalls for the full transform_timeout.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        # optionally append /compressed to detection and segmentation topics if not in the strings
        if self.input_image_topic_is_compressed:
            if not self.input_image_topic.endswith("/compressed"):
                self.input_image_topic = self.input_image_topic + "/compressed"
            if not self.detection_image_topic.endswith("/compressed"):
                self.detection_image_topic = self.detection_image_topic + "/compressed"
            if not self.segmentation_image_topic.endswith("/compressed"):
                self.segmentation_image_topic = self.segmentation_image_topic + "/compressed"
            if not self.segmentation_mask_image_topic.endswith("/compressed"):
                self.segmentation_mask_image_topic = self.segmentation_mask_image_topic + "/compressed"

        # setup QoS
        qos_profile = QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.queue_size
        )
        if self.qos.lower() == "sensor_data":
            qos_profile = QoSProfile(
                    reliability=QoSReliabilityPolicy.BEST_EFFORT,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=self.queue_size
            )

        self.image_message_format = "raw"
        self.image_message_type = Image
        if self.input_image_topic_is_compressed or "compressed" in self.input_image_topic:
            self.image_message_format = "compressed"
            self.image_message_type = CompressedImage

        # Setup inference dictionary
        self.inference_dict = {
            'source': None,
            'conf': self.conf_thresh,
            'iou': self.iou_thresh,
            'imgsz': self.imgsz,
            'device': self.device,
            'half': self.half_precision,
            'classes': self.classes,
            'max_det': self.max_det,
            'retina_masks': True,
            'show': False,
            'stream': False,
            'augment': self.augment,
            'agnostic_nms': self.agnostic_nms,
            'verbose': self.verbose
        }

        # (optional) modify tracker parameters
        if self.track_2d:
            with open(self.tracker_2d_cfg['path'], 'r') as file:
                tracker_config = yaml.safe_load(file)

            for k, new_v in self.tracker_2d_cfg.copy().items():
                if k in tracker_config.keys():
                    tracker_config[k] = update_tracker_param(k, new_value=new_v, old_value=tracker_config[k])

            assert tracker_config['tracker_type'] in [
                "bytetrack",
                "botsort",
            ], f"Only 'bytetrack' and 'botsort' are supported for now, but got '{tracker_config['tracker_type']}'"

            # Create a temporary file
            self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
            yaml.safe_dump(tracker_config, self.temp_file, default_flow_style=False, sort_keys=False)

            self.tracker_2d_cfg['path'] = self.temp_file.name

        # Optional per-stage execution-time profiling (off by default).
        self.profiler = setup_profiler(self)

        # Setup dynamic parameter reconfiguring.
        # Register a callback function that will be called whenever there is an attempt to
        # change one or more parameters of the node.
        self.add_on_set_parameters_callback(self.parameter_change_callback)

        # Subscribers
        self.subscriptions_ = []
        self.image_sub = Subscriber(self, self.image_message_type, self.input_image_topic, qos_profile=qos_profile)
        self.subscriptions_.append(self.image_sub)
        self.input_camera_info_sub = Subscriber(self, CameraInfo, self.input_camera_info_topic, qos_profile=qos_profile)
        self.subscriptions_.append(self.input_camera_info_sub)

        if self.project_to_3d:
            if self.use_depth:
                self.depth_sub = Subscriber(self, Image, self.depth_image_topic, qos_profile=qos_profile)
                self.subscriptions_.append(self.depth_sub)
                self.depth_camera_info_sub = Subscriber(self, CameraInfo, self.depth_camera_info_topic,
                                                        qos_profile=qos_profile)
                self.subscriptions_.append(self.depth_camera_info_sub)

            if self.use_pointcloud:
                self.pointcloud_sub = Subscriber(self, PointCloud2, self.pointcloud_topic, qos_profile=qos_profile)
                self.subscriptions_.append(self.pointcloud_sub)

        if self.synchronization_interval > 0.0:
            self.ts = ApproximateTimeSynchronizer(self.subscriptions_, self.queue_size,
                                                  slop=self.synchronization_interval)
        else:
            self.ts = TimeSynchronizer(self.subscriptions_, self.queue_size)
        self.ts.registerCallback(self.detection_callback)

        # Publishers
        # self.cluster_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)
        self.detection_results_pub = self.create_publisher(Detection2DArray, self.detection_results_topic,
                                                           self.queue_size)

        if self.publish_object_array:
            self.object_array_pub = self.create_publisher(
                    ObjectArray,
                    'yolo/objects',
                    self.queue_size
            )

        self._nav2_warned = False
        if self.publish_obstacle_array:
            if NAV2_DYNAMIC_MSGS_AVAILABLE:
                self.obstacle_detection_pub = self.create_publisher(ObstacleArray, 'yolo/obstacles', qos_profile)
            else:
                self.get_logger().warning("nav2_dynamic_msgs not available; nav2 ObstacleArray output disabled")
                self._nav2_warned = True

        if self.publish_debug_image:
            if self.detection_image_topic:
                self.detection_image_pub = self.create_publisher(self.image_message_type,
                                                                 self.detection_image_topic, self.queue_size)

            if self.segmentation_image_topic:
                self.segmentation_image_pub = self.create_publisher(self.image_message_type,
                                                                    self.segmentation_image_topic, self.queue_size)

            if self.segmentation_mask_image_topic:
                self.segmentation_mask_image_pub = self.create_publisher(self.image_message_type,
                                                                         self.segmentation_mask_image_topic, self.queue_size)
        if self.project_to_3d:
            if self.use_depth:
                self.detection3d_depth_results_pub = self.create_publisher(Detection3DArray,
                                                                           "yolo/detection3d_depth_results",
                                                                           self.queue_size)
                self.marker_depth_pub = self.create_publisher(MarkerArray,
                                                              'yolo/detection3d_depth_markers',
                                                              self.queue_size)
            if self.use_pointcloud:
                self.detection3d_pointcloud_results_pub = self.create_publisher(Detection3DArray,
                                                                           "yolo/detection3d_pointcloud_results",
                                                                           self.queue_size)
                self.marker_pointcloud_pub = self.create_publisher(MarkerArray,
                                                              'yolo/detection3d_pointcloud_markers',
                                                              self.queue_size)

        # # Timers
        # self.timer = self.create_timer(0.1, self.timer_callback)
        # self.timer_count = 0
        self.get_logger().info(f"image_obstacle_detection_node node started on device: {self.device}")


    def detection_callback(self, *msg):
        """
        Todo: If the input to YOLO is not a Pytorch Tensor, the array is converted to RGB using numpy.
        Therefore, manually transfer the image to a torch tensor and perform operations (e.g color conversion and resizing) using torch before passing to YOLO to reduce CPU utilization. """
        try:
            msg_timestamp = None
            msg_fmt = "bgr8"
            conversion = None
            inverse_conversion = None
            is_color = True
            is_depth = False
            self.camera_info_callback(msg)
            for camera in self.cameras:
                img_msg = msg[0] if camera == 'rgb' else msg[2]
                cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt, conversion, inverse_conversion, is_color, is_depth, compressed_msg_codec = _parse_image_message(
                    img_msg, self.bridge, self.image_message_format, depth_scale=self.depth_scale, logger=self.get_logger())
                self.frame_ids[camera] = image_frame_id
                self.headers[camera] = img_msg.header
                self.msg_metadata[camera] = {'msg_encoding': msg_encoding, 'msg_timestamp': msg_timestamp,
                                             'msg_fmt': msg_fmt, 'conversion': conversion,
                                             'inverse_conversion': inverse_conversion,
                                             'is_color': is_color, 'is_depth': is_depth,
                                             'compressed_msg_codec': compressed_msg_codec}

                if camera == 'rgb':
                    # get image dimensions
                    if self.imgsz is None:
                        self.image_height, self.image_width = cv_image.shape[:2]
                        if self.use_image_dimensions:
                            # Check divisibility by 32 to conform to YOLOs convolution kernel size and stride length
                            new_height = self.image_height if self.image_height % 32 == 0 else ((self.image_height // 32) + 1) * 32
                            new_width = self.image_width if self.image_width % 32 == 0 else ((self.image_width // 32) + 1) * 32

                            self.imgsz = (new_height, new_width)
                            self.get_logger().info(f"Using image dimensions: {self.imgsz}")
                        else:
                            self.imgsz = list(self.image_dimensions)  # [640, 640]

                    # (optional) resize the image
                    if self.resize_image and (
                            (self.image_height, self.image_width) != (self.imgsz[0], self.imgsz[1])):
                        cv_image = cv2.resize(cv_image,
                                              (self.imgsz[1], self.imgsz[0]))  # , interpolation=cv2.INTER_LINEAR

                # update the image dictionary
                self.images[camera] = cv_image

            self.inference_dict['imgsz'] = self.imgsz
            # detect/track objects in the visual image
            with self.profiler.measure("detection"):
                self.detect_objects(self.images['rgb'])
            self.profiler.record_speed(self.results)

            if self.project_to_3d and self.use_pointcloud:
                # unpack pointcloud message
                ros_cloud = msg[-1]
                # extract PointCloud from struct message
                self.pointcloud_preprocessor.extract_pointcloud(ros_cloud)

                # Preprocess pointcloud: if input is a pointcloud, transform the pointcloud here
                self.frame_ids['pointcloud'] = ros_cloud.header.frame_id
                self.headers['pointcloud'] = ros_cloud.header
                self.msg_metadata['pointcloud'] = {
                    'msg_timestamp': ros_cloud.header.stamp,  # None
                }

                # get transform to robot frame
                self.get_camera_to_robot_tf(
                        self.frame_ids['pointcloud'],
                        None if self.static_camera_to_robot_tf else self.msg_metadata['pointcloud'].get('msg_timestamp'),
                )
                self.pointcloud_preprocessor.camera_to_robot_tf = self.camera_to_robot_tf_o3d

                # preprocess the pointcloud
                self.pointcloud_preprocessor.preprocess()
                new_header = self.pointcloud_preprocessor.create_header(ros_cloud)
                pc_fields = self.pointcloud_preprocessor.pointcloud_metadata['field_names']
                self.msg_metadata['pointcloud']['field_names'] = pc_fields

                # copy the pointcloud.  todo: use one instead of all(self.o3d_pointcloud, self.pointcloud_preprocessor.o3d_pointcloud, self.images['pointcloud'])
                self.o3d_pointcloud = self.pointcloud_preprocessor.o3d_pointcloud.clone()
                # todo: transform the pointcloud in case the PointCloud Preprocessor node fails to transform
                # if self.camera_to_robot_tf_o3d is not None:
                #     # copy the original positions before transforming for later use without inversion.
                #     # Leads to significant speedup over multiple transformations.
                #     self.o3d_pointcloud.point.positions_inv = self.o3d_pointcloud.point.positions.clone()
                #     frame_id = self.output_frame

                if not self.o3d_pointcloud.is_empty():
                    self.images['pointcloud'] = self.o3d_pointcloud.point.positions  # .cpu().numpy()  # todo: refactor without transfering to CPU or using numpy, i.e use torch tensors

            # try:
            #    self.results = next(self.results)
            # except TypeError:
            #    pass
            detection_msg, detection_image, mask_img = self.parse_results(self.results, self.headers['rgb'])

            if detection_msg is None:
                return

            # publish the detection array results
            self.detection_results_pub.publish(detection_msg)

            # convert OpenCV image back to the input msg_fmt
            if detection_image is not None:
                if self.publish_debug_image:
                    if self.msg_metadata['rgb'].get('conversion', None) is not None:
                        detection_image = cv2.cvtColor(detection_image, self.msg_metadata['rgb'].get('inverse_conversion'))

                    if self.image_message_format in ("compressed", "packet"):
                        detection_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                                detection_image,
                                dst_format=self.msg_metadata['rgb'].get('compressed_msg_codec'))  # msg.format.split(';')[1].split()[0]
                    else:
                        detection_image_msg = self.bridge.cv2_to_imgmsg(
                                detection_image,
                                encoding=self.msg_metadata['rgb'].get('msg_fmt'))

                    detection_image_msg.header.frame_id = self.frame_ids['rgb']
                    detection_image_msg.header.stamp = self.headers['rgb'].stamp

                    if self.detection_image_topic:
                        self.detection_image_pub.publish(detection_image_msg)
                    if self.segmentation_mask_image_topic and (mask_img is not None):
                        if self.image_message_format in ("compressed", "packet"):
                            mask_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                                    mask_img,
                                    dst_format=self.msg_metadata['rgb'].get('compressed_msg_codec'))  # msg.format.split(';')[1].split()[0]
                        else:
                            mask_image_msg = self.bridge.cv2_to_imgmsg(
                                    mask_img,
                                    encoding="mono8")

                        mask_image_msg.header.frame_id = self.frame_ids['rgb']
                        mask_image_msg.header.stamp = self.headers['rgb'].stamp
                        self.segmentation_mask_image_pub.publish(mask_image_msg)

                    if self.segmentation_image_topic and (mask_img is not None):
                        # color_mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                        cv_image_inverted = self.images['rgb']
                        if self.msg_metadata['rgb'].get('inverse_conversion') is not None:
                            cv_image_inverted = cv2.cvtColor(self.images['rgb'],
                                                             self.msg_metadata['rgb'].get('inverse_conversion'))
                        if (mask_img.shape[1], mask_img.shape[0]) != (cv_image_inverted.shape[1], cv_image_inverted.shape[0]):
                            mask_img = cv2.resize(mask_img, (cv_image_inverted.shape[1], cv_image_inverted.shape[0]), interpolation=cv2.INTER_NEAREST)
                        color_mask_img = cv2.bitwise_and(cv_image_inverted, cv_image_inverted, mask=mask_img)
                        if self.show_image:
                            try:
                                cv2.imshow("color_mask_image", color_mask_img)
                                cv2.waitKey(1)
                            except Exception as e:
                                self.get_logger().warning(
                                    f"Could not display window 'color_mask_image' (likely headless environment): {e}. Disabling show_image."
                                )
                                self.show_image = False

                        if self.image_message_format in ("compressed", "packet"):
                            color_mask_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                                color_mask_img,
                                dst_format=self.msg_metadata['rgb'].get(
                                    'compressed_msg_codec'))  # msg.format.split(';')[1].split()[0]
                        else:
                            color_mask_image_msg = self.bridge.cv2_to_imgmsg(
                                color_mask_img,
                                encoding=self.msg_metadata['rgb'].get('msg_fmt', 'bgr8'))  # passthrough

                        color_mask_image_msg.header.frame_id = self.frame_ids['rgb']
                        color_mask_image_msg.header.stamp = self.headers['rgb'].stamp
                        self.segmentation_image_pub.publish(color_mask_image_msg)

            self.profiler.flush(self)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            raise e
            # if self.debug:
            #     raise e


    def camera_info_callback(self, msg):
        # Save/update camera infos
        for camera in self.cameras:
            # initialize the camera infos and models
            if self.camera_infos[camera] is None:
                self.camera_infos[camera] = msg[1] if camera == 'rgb' else msg[3]
                self.camera_models[camera].fromCameraInfo(self.camera_infos[camera])
                if self.use_pointcloud:
                    self.o3d_camera_intrinsics[camera] = self.convert_to_open3d_tensor(self.camera_models[camera].K)

            # update the camera infos and models if not static
            if not self.static_camera_info:
                self.camera_infos[camera] = msg[1] if camera == 'rgb' else msg[3]
                self.camera_models[camera].fromCameraInfo(self.camera_infos[camera])
                if self.use_pointcloud:
                    self.o3d_camera_intrinsics[camera] = self.convert_to_open3d_tensor(self.camera_models[camera].K)

    def unpack_pointcloud_message_old(self, ros_cloud):
        # todo: remove
        frame_id = ros_cloud.header.frame_id
        msg_timestamp = ros_cloud.header.stamp
        field_names = ('x', 'y', 'z', 'rgb')

        try:
            # Get field indices for faster access
            xyz_offset = [None, None, None]
            rgb_offset = None

            for idx, field in enumerate(ros_cloud.fields):
                # todo: add all fields
                if field.name == 'x':
                    xyz_offset[0] = idx
                elif field.name == 'y':
                    xyz_offset[1] = idx
                elif field.name == 'z':
                    xyz_offset[2] = idx
                elif field.name == 'rgb':
                    rgb_offset = idx

            if None in xyz_offset or rgb_offset is None:
                self.get_logger().error("Required point cloud fields not found")
                return None, None, None

            # Convert ROS PointCloud2 to numpy arrays
            # https://github.com/ros2/common_interfaces/blob/humble/sensor_msgs_py/sensor_msgs_py/point_cloud2.py
            # https://gist.github.com/SebastianGrans/6ae5cab66e453a14a859b66cd9579239?permalink_comment_id=4345802#gistcomment-4345802
            cloud_array = point_cloud2.read_points_numpy(
                ros_cloud,
                field_names=field_names,
                skip_nans=True
            )

            # Extract XYZ points
            points_np = cloud_array[:, :3].astype(np.float32)

            # Extract and convert RGB values
            rgb_float = cloud_array[:, 3].copy()
            rgb_bytes = rgb_float.view(np.uint32)

            # Extract RGB channels
            r = ((rgb_bytes >> 16) & 0xFF).astype(np.float32) / 255.0
            g = ((rgb_bytes >> 8) & 0xFF).astype(np.float32) / 255.0
            b = (rgb_bytes & 0xFF).astype(np.float32) / 255.0

            # Stack RGB channels
            colors_np = np.vstack((r, g, b)).T  # todo:use hstack instead

            # Convert numpy arrays to tensors and move to device
            self.o3d_pointcloud.point.positions = o3d.core.Tensor(
                points_np,
                dtype=o3d.core.Dtype.Float32,
                device=self.o3d_device
            )

            self.o3d_pointcloud.point.colors = o3d.core.Tensor(
                colors_np,
                dtype=o3d.core.Dtype.Float32,
                device=self.o3d_device
            )
            return frame_id, msg_timestamp, field_names
        except Exception as e:
            self.get_logger().error(f"Error processing point cloud: {str(e)}")
            # raise e
            return None, None, None

    def detect_objects(self, image):
        try:
            self.inference_dict['source'] = image
            if self.track_2d:
                # https://docs.ultralytics.com/modes/track/#why-choose-ultralytics-yolo-for-object-tracking
                self.results = self.model.track(
                        tracker=self.tracker_2d_cfg['path'],
                        persist=True,
                        **self.inference_dict
                )
            else:
                # https://docs.ultralytics.com/modes/predict/#inference-arguments
                self.results = self.model.predict(**self.inference_dict)

                # # or control each step (predict/track does all three steps)
                # im = model.predictor.preprocess(source)[0]
                # preds = model.predictor.model(im)  # inference only
                # results = model.predictor.postprocess(preds)  # post-processing
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            self.results = None

    def parse_results(self, results, header):
        if results is not None:
            detections_msg, mask_img = self.create_detections_array(results, header)

            return detections_msg, self.detection_image, mask_img
        return None, None, None

    def create_detections_array(self, results, header):
        # Create 2D result messages
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = self.msg_metadata['rgb'].get('msg_timestamp')  # self.get_clock().now().to_msg()
        detections_msg.header.frame_id = self.frame_ids['rgb']

        objects_msg = ObjectArray()
        objects_msg.header.stamp = self.msg_metadata['rgb'].get('msg_timestamp')   # self.get_clock().now().to_msg()
        objects_msg.header.frame_id = self.frame_ids['rgb']

        if self.publish_obstacle_array and NAV2_DYNAMIC_MSGS_AVAILABLE:
            obstacle_msg = ObstacleArray()
            obstacle_msg.header.stamp = self.msg_metadata['rgb'].get('msg_timestamp')  # self.get_clock().now().to_msg()
            obstacle_msg.header.frame_id = self.frame_ids['rgb']
        else:
            obstacle_msg = None

        mask_img = None

        if self.project_to_3d:
            # get transform
            if self.use_depth:
                depth_timestamp = self.msg_metadata['depth'].get(
                    'msg_timestamp') if self.output_frame else self.msg_metadata['rgb'].get(
                    'msg_timestamp')  # self.get_clock().now().to_msg()

                if depth_timestamp is None:
                    depth_timestamp = self.get_clock().now().to_msg()

                # Cache the depth-camera -> output_frame transform used to lift
                # optical-frame boxes into the output frame. Deliberately not
                # camera_to_robot_tf: the pointcloud branch caches the lidar frame
                # there and static caching would keep whichever was set first.
                if self.output_frame and (
                        self.depth_to_output_tf is None or not self.static_camera_to_robot_tf):
                    _tf = self.lookup_transform(
                            self.frame_ids['depth'], self.output_frame,
                            None if self.static_camera_to_robot_tf else depth_timestamp)
                    if _tf is not None:
                        self.depth_to_output_tf = self.transform_to_matrix(_tf)
                detection3d_depth_array = Detection3DArray()
                detection3d_depth_array.header.frame_id = self.output_frame if self.output_frame else self.frame_ids['depth']
                detection3d_depth_array.header.stamp = depth_timestamp

                marker_depth_array = MarkerArray()

            if self.use_pointcloud:
                detection3d_pointcloud_array = Detection3DArray()
                detection3d_pointcloud_array.header.frame_id = self.output_frame if self.output_frame else self.frame_ids[
                    'pointcloud']
                detection3d_pointcloud_array.header.stamp = self.msg_metadata['pointcloud'].get(
                    'msg_timestamp') if self.output_frame else self.msg_metadata['rgb'].get(
                    'msg_timestamp')  # self.get_clock().now().to_msg()

                marker_pointcloud_array = MarkerArray()


        for result in results:
            self.detection_image = result.plot(
                    conf=True,
                    labels=True,
                    boxes=True,
                    masks=True,
                    probs=True,
                    # # todo: use the image below to specify the original image if passing an ROI masked image to the detector
                    # img=self.inference_dict['source'] or self.images['rgb'],  # numpy image to overlay detections on. This is slower since it needs to be transferred to GPU
                    # im_gpu=None,  # torch tensor image to overlay detections on. This is faster since it does not need to be transferred to GPU
            )
            if self.show_image:
                try:
                    # Visualize the results on the frame
                    cv2.imshow("image", self.detection_image)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warning(
                        f"Could not display window 'image' (likely headless environment): {e}. Disabling show_image."
                    )
                    self.show_image = False

            # use result.cpu().numpy()  # to move all at once. or result.to(device="cpu", dtype=torch.float32)
            result = result.cpu()
            # todo: do not hardcode cpu usage as we can postprocess with depth/pointcloud on GPU. Remove cpu() and numpy() calls

            bounding_box = result.boxes.cpu()  # Boxes object for bounding box outputs. n x 4
            classes = result.boxes.cls.cpu()  # n,
            confidence_score = result.boxes.conf.cpu()  # n,
            masks = result.masks
            keypoints = result.keypoints
            obb = result.obb
            probs = result.probs

            if bounding_box.shape[0] < 1:
                # Zero detections: skip projection but emit a heartbeat (empty 3D +
                # DELETEALL markers) so downstream costmaps clear and stale RViz
                # markers are removed when objects leave the frame.
                if self.project_to_3d and self.publish_empty_detections:
                    if self.use_depth:
                        self.detection3d_depth_results_pub.publish(detection3d_depth_array)
                        self.marker_depth_pub.publish(make_deleteall_marker_array())
                    if self.use_pointcloud:
                        self.detection3d_pointcloud_results_pub.publish(detection3d_pointcloud_array)
                        self.marker_pointcloud_pub.publish(make_deleteall_marker_array())
                return detections_msg, mask_img

            track_ids = None
            if self.track_2d and bounding_box.is_track:
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()

            if hasattr(result, "masks") and masks is not None:
                # mask_data = masks.data.cpu()  # masks drawn on the image [0-1] float. n x image_height x image_width
                # mask_orig_shape = masks.orig_shape  # tuple [image_width, image_height]
                # masks_shape = masks.shape
                # masks_xy = masks.xy  # list of size n, each item (ndarray) of size [m, 2], where m is the number of pixels per object mask
                mask_img = (torch.sum(masks.data, dim=0).cpu().numpy() * 255).astype(np.uint8)
                if self.show_image:
                    try:
                        cv2.imshow("masked_image", mask_img)
                        cv2.waitKey(1)
                    except Exception as e:
                        self.get_logger().warning(
                            f"Could not display window 'masked_image' (likely headless environment): {e}. Disabling show_image."
                        )
                        self.show_image = False

            else:
                # create a list of Nones
                masks = [None] * len(bounding_box)  # bounding_box.shape[0]

            if keypoints is not None:
                keypoints = keypoints.cpu()  # Keypoints object for pose outputs

            if obb is not None:
                obb = obb.cpu()  # Oriented boxes object for OBB outputs

            if probs is not None:
                probs = probs.cpu()  # Probs object for classification outputs

            for i, (box, cls, conf, mask) in enumerate(zip(bounding_box, classes, confidence_score, masks)):
                # todo: speed up by avoiding this for-loop, e.g pass the bounding_boxes, masks and (depth/pointcloud) to the detection for loop

                # preprocess
                bbox = box.xywh.cpu().numpy().flatten()
                x, y, w, h = bbox
                if mask is not None:
                    # mask = mask.cpu().numpy()
                    mask_xy = mask.xy[0]
                track_id = box.id.int().cpu().item() if box.id is not None else -1

                if self.track_2d and self.plot_tracks:
                    track = self.track_history[track_id]
                    track.append((float(x), float(y)))  # x, y center point
                    if len(track) > 30:  # retain 30 tracks for 30 frames
                        track.pop(0)

                    # Draw the tracking lines
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(self.detection_image, [points],
                                  isClosed=False, color=(230, 230, 230), thickness=5)

                # pack 2D detection results
                detection_2d = pack_2d_detection(
                    bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)), conf,
                        id=track_ids[i] if track_ids is not None else -1)
                detections_msg.detections.append(detection_2d)

                # pack object message
                object_2d = pack_derived_object_msg(bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)),
                                                    conf,
                                                    id=track_ids[i] if track_ids is not None else -1)
                objects_msg.objects.append(object_2d)

                if obstacle_msg is not None:
                    obstacle_2d = pack_nav2_obstacle_msg(
                        bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)), conf, id=track_ids[i] if track_ids is not None else -1, z_size=0.0)  # todo: replace with height
                    obstacle_msg.obstacles.append(obstacle_2d)

                if self.project_to_3d:
                    # could run depth and pointcloud processing in different threads
                    if self.use_depth and (self.images['depth'] is not None):
                        with self.profiler.measure("depth_projection"):
                            x, y, z, size_x, size_y, size_z, quat, points_3d = self.project_to_3d_with_depth(
                                mask, bbox, self.images['depth'], infer_shape=result.orig_shape)

                        if x is not None:
                            # transform the boxes from the camera optical frame
                            # (x-right, y-down, z-forward) to the robot frame
                            frame_id = self.frame_ids["depth"]
                            depth_box_valid = True
                            if self.output_frame:
                                frame_id = self.output_frame
                                if self.depth_to_output_tf is not None:
                                    x, y, z, size_x, size_y, size_z = self.transform_box_to_frame(
                                        x, y, z, size_x, size_y, size_z, self.depth_to_output_tf)
                                else:
                                    # publishing optical coords stamped as output_frame
                                    # would place boxes above/behind the robot
                                    depth_box_valid = False
                                    self.get_logger().warn(
                                        f"No TF {self.frame_ids['depth']} -> {self.output_frame} yet; "
                                        f"skipping depth 3D detection.", throttle_duration_sec=5.0)

                            # Append 3D detection (x, y, z, size_x, size_y, size_z, confidence, class_id)
                            # frame_id: output_frame (base_link) when set, else depth sensor frame
                            if depth_box_valid:
                                detection3d_depth_array.detections.append(
                                    self.create_3d_detection(x, y, z, size_x, size_y, size_z, conf,
                                                             result.names.get(int(cls)), frame_id=frame_id))
                                marker_depth_array.markers.append(
                                    self.create_marker(
                                        i, x, y, z, size_x, size_y, size_z, frame_id,
                                        depth_timestamp, conf, result.names.get(int(cls)), track_id=None,
                                        rgba=[1.0, 0.0, 0.0, 0.5]))

                    if self.use_pointcloud and (self.images['pointcloud'] is not None):
                        with self.profiler.measure("pointcloud_projection"):
                            x, y, z, size_x, size_y, size_z, quat, points_3d = self.project_to_3d_with_pointcloud(
                                mask, bbox, self.images['pointcloud'])

                        if x is not None:
                            # no need to transform the boxes to the robot frame since the pointcloud is transformed
                            frame_id = self.frame_ids["pointcloud"] if not self.output_frame else self.output_frame

                            # Append 3D detection (x, y, z, size_x, size_y, size_z, confidence, class_id)
                            detection3d_pointcloud_array.detections.append(
                                self.create_3d_detection(
                                    x, y, z, size_x, size_y, size_z, conf, result.names.get(int(cls)), quat,
                                    frame_id=frame_id)
                            )
                            marker_pointcloud_array.markers.append(
                                self.create_marker(
                                    i, x, y, z, size_x, size_y, size_z, frame_id,
                                    self.msg_metadata['pointcloud'].get('msg_timestamp'), conf, result.names.get(int(cls)), track_id=None, quat=quat,
                                    rgba=[0.0, 1.0, 0.0, 0.5]))

            # publish messages
            if self.publish_object_array and hasattr(self, 'object_array_pub'):
                self.object_array_pub.publish(objects_msg)

            if obstacle_msg is not None and hasattr(self, 'obstacle_detection_pub'):
                self.obstacle_detection_pub.publish(obstacle_msg)

            if self.project_to_3d:
                if self.use_depth:
                    self.detection3d_depth_results_pub.publish(detection3d_depth_array)
                    # No valid projection this frame: DELETEALL clears stale markers
                    # (an empty MarkerArray would leave old boxes on screen).
                    if not marker_depth_array.markers and self.publish_empty_detections:
                        self.marker_depth_pub.publish(make_deleteall_marker_array())
                    else:
                        self.marker_depth_pub.publish(marker_depth_array)

                if self.use_pointcloud:
                    self.detection3d_pointcloud_results_pub.publish(detection3d_pointcloud_array)
                    if not marker_pointcloud_array.markers and self.publish_empty_detections:
                        self.marker_pointcloud_pub.publish(make_deleteall_marker_array())
                    else:
                        self.marker_pointcloud_pub.publish(marker_pointcloud_array)
        return detections_msg, mask_img

    @staticmethod
    def align_depth_to_rgb(depth_map, K_depth, K_rgb, T_depth_to_rgb, rgb_shape, depth_scale=1.0):
        """
        Aligns a depth map to the perspective of an RGB camera using PyTorch.

        Args:
            depth_map (torch.Tensor): Source depth image (H_d, W_d).
            K_depth (torch.Tensor): Depth camera intrinsics (3, 3).
            K_rgb (torch.Tensor): RGB camera intrinsics (3, 3).
            T_depth_to_rgb (torch.Tensor): Transformation matrix (4, 4) from Depth to RGB.
            rgb_shape (tuple): Target resolution (H_rgb, W_rgb).

        Returns:
            torch.Tensor: Aligned depth map matching the RGB resolution (H_rgb, W_rgb).
        """
        H_d, W_d = depth_map.shape
        H_rgb, W_rgb = rgb_shape
        device = depth_map.device

        # 1. Create a grid of coordinates for the depth image
        v, u = torch.meshgrid(torch.arange(H_d, device=device), torch.arange(W_d, device=device), indexing='ij')

        # Flatten everything for vectorized operations
        u = u.flatten()
        v = v.flatten()
        z = depth_map.flatten() / depth_scale

        # Filter out invalid depth points (zeros or negative)
        valid_mask = z > 0
        u, v, z = u[valid_mask], v[valid_mask], z[valid_mask]

        # 2. Unproject to 3D in Depth Coordinate System
        # x = (u - cx) * z / fx
        # y = (v - cy) * z / fy
        cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]
        fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]

        x_3d = (u - cx_d) * z / fx_d
        y_3d = (v - cy_d) * z / fy_d
        ones = torch.ones_like(z)

        # Stack to create homogeneous coordinates (4, N)
        points_3d_depth = torch.stack([x_3d, y_3d, z, ones], dim=0)

        # 3. Transform to RGB Coordinate System
        # Apply the extrinsic matrix (R | T)
        points_3d_rgb = T_depth_to_rgb @ points_3d_depth  # Matrix multiplication

        # Extract transformed coordinates
        x_rgb = points_3d_rgb[0, :]
        y_rgb = points_3d_rgb[1, :]
        z_rgb = points_3d_rgb[2, :]

        # Filter points that are behind the RGB camera (z <= 0)
        valid_z_mask = z_rgb > 0
        x_rgb, y_rgb, z_rgb = x_rgb[valid_z_mask], y_rgb[valid_z_mask], z_rgb[valid_z_mask]

        # 4. Project to 2D RGB Plane
        # u' = (x' * fx' / z') + cx'
        cx_rgb, cy_rgb = K_rgb[0, 2], K_rgb[1, 2]
        fx_rgb, fy_rgb = K_rgb[0, 0], K_rgb[1, 1]

        u_proj = (x_rgb * fx_rgb / z_rgb) + cx_rgb
        v_proj = (y_rgb * fy_rgb / z_rgb) + cy_rgb

        # Round to nearest integer pixel coordinates
        u_proj = torch.round(u_proj).long()
        v_proj = torch.round(v_proj).long()

        # 5. Handle Bounds and Occlusions
        # Filter points falling outside the RGB image resolution
        in_bounds = (u_proj >= 0) & (u_proj < W_rgb) & (v_proj >= 0) & (v_proj < H_rgb)

        u_final = u_proj[in_bounds]
        v_final = v_proj[in_bounds]
        z_final = z_rgb[in_bounds]

        # Occlusion handling: Painter's Algorithm
        # Sort by depth (descending) so closer points (processed last) overwrite further points
        sorted_indices = torch.argsort(z_final, descending=True)
        u_final = u_final[sorted_indices]
        v_final = v_final[sorted_indices]
        z_final = z_final[sorted_indices]

        # Initialize output canvas
        aligned_depth = torch.zeros((H_rgb, W_rgb), device=device, dtype=torch.float32)

        # Assign values to the canvas
        aligned_depth[v_final, u_final] = z_final

        return aligned_depth * depth_scale

    def project_to_3d_with_depth(self, mask, xywh, depth_image, infer_shape=None):
        # todo: refactor as this is wrong and does not work, e.g with carla
        """
        Steps:
            1. (optional) resize mask data (and xy) to the depth image size
            2. Get the ROI of the mask (or bbox) in the depth image
            3. Scale the roi depth mask, e.g to convert to meters
            4. filter out invalid depth pixels, i.e roi[roi > 0]
            5. Find the z coordinate (distance) of the mask, np.median(roi) or bounding_box_center/depth_scale
            6. (optional) crop values outside of depth_max
            7. Project from image to world space
        :param mask:
        :param xywh:
        :param depth_image:
        :return:
        """
        # Ultralytics boxes live in the source-image pixel space (result.orig_shape,
        # e.g. 640x640 when resize_image shrank the frame before inference); rescale
        # to the depth image pixel space before indexing it or unprojecting with
        # intrinsics.
        ref_h, ref_w = infer_shape[:2] if infer_shape else depth_image.shape[:2]
        u_scale = depth_image.shape[1] / float(ref_w)
        v_scale = depth_image.shape[0] / float(ref_h)
        bbox_center_x, bbox_center_y = int(xywh[0] * u_scale), int(xywh[1] * v_scale)
        bbox_size_x, bbox_size_y = int(xywh[2] * u_scale), int(xywh[3] * v_scale)

        # Step 1: Project the depth image to the RGB frame.
        dtype = torch.float32
        H_rgb, W_rgb = self.camera_models["rgb"].height, self.camera_models["rgb"].width  # or get from the image
        cx_rgb, cy_rgb = self.camera_models["rgb"].cx(), self.camera_models["rgb"].cy()
        fx_rgb, fy_rgb = self.camera_models["rgb"].fx(), self.camera_models['rgb'].fy()
        k_rgb = self.camera_models["rgb"].K  # self.camera_infos['rgb'].k.reshape(3,3)

        H_depth, W_depth = self.camera_models["depth"].height, self.camera_models["depth"].width  # or get from the image
        cx_depth, cy_depth = self.camera_models["depth"].cx(), self.camera_models["depth"].cy()
        fx_depth, fy_depth = self.camera_models["depth"].fx(), self.camera_models['depth'].fy()
        k_depth = self.camera_models["depth"].K  # self.camera_infos['depth'].k.reshape(3,3)

        # transform points to RGB
        if self.depth_to_rgb_tf is None or not self.static_camera_to_robot_tf:
            source_frame = self.frame_ids['depth']

            transform = self.lookup_transform(source_frame, self.frame_ids['rgb'], rclpy.time.Time())

            if transform is not None:
                self.depth_to_rgb_tf = self.transform_to_matrix(transform)
                self.depth_to_rgb_tf_torch = torch.as_tensor(self.depth_to_rgb_tf, dtype=dtype, device=self.torch_device)

        if (self.depth_to_rgb_tf_torch is not None) and not torch.equal(
                self.depth_to_rgb_tf_torch,
                torch.eye(self.depth_to_rgb_tf_torch.shape[0], dtype=dtype, device=self.torch_device)):
            depth_image = self.align_depth_to_rgb(
                    torch.from_numpy(depth_image).to(dtype=dtype, device=self.torch_device),
                    torch.from_numpy(k_depth).to(dtype=dtype, device=self.torch_device),
                    torch.from_numpy(k_rgb).to(dtype=dtype, device=self.torch_device),
                    self.depth_to_rgb_tf_torch, (H_rgb, W_rgb), self.depth_scale)
            depth_image = depth_image.cpu().numpy()

        # Step 2: Get the ROI of the mask (or bbox) in the depth image
        if mask is not None:
            mask_data = mask.data.cpu().numpy().astype(np.uint8)[0, :, :] * 255
            mask_xy = mask.xy[0]

            # crop depth image by mask
            # mask_array = np.array(
            #     [[int(ele[0]), int(ele[1])] for ele in mask_xy]
            # )
            # mask_ = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            # cv2.fillPoly(mask_, [np.array(mask_array, dtype=np.int32)], 255)
            # roi = cv2.bitwise_and(depth_image, depth_image, mask=mask_)  # same as below

            if (mask_data.shape[1], mask_data.shape[0]) != (depth_image.shape[1], depth_image.shape[0]):
                mask_data = cv2.resize(mask_data, (depth_image.shape[1], depth_image.shape[0]), interpolation=cv2.INTER_NEAREST)

            roi = cv2.bitwise_and(depth_image, depth_image, mask=mask_data)  # same as above

        else:
            # crop depth image by the 2d BB. todo: use xyxy
            u_min = int(max(bbox_center_x - bbox_size_x // 2, 0))
            u_max = int(min(bbox_center_x + bbox_size_x // 2, depth_image.shape[1] - 1))
            v_min = int(max(bbox_center_y - bbox_size_y // 2, 0))
            v_max = int(min(bbox_center_y + bbox_size_y // 2, depth_image.shape[0] - 1))

            roi = depth_image[v_min:v_max, u_min:u_max]

        '''
        Method 2:
        mask_depth_values= depth_image[mask_data > 0]   # depth_values: same
        if mask_depth_values.size == 0:
            return

        Method 3:
        # mask_indices = np.argwhere(mask_data> 0)
        # mask_depth_values = depth_image[mask_indices[:, 0], mask_indices[:, 1]]   # depth_values: same
        # if mask_depth_values.size == 0:
        #     return

        # Filter valid depth values
        valid_depths = mask_depth_values[np.isfinite(mask_depth_values)]   # depth_values: same
        if valid_depths.size == 0:
            return
        '''

        # Step 3: Scale the roi depth mask, e.g to convert to meters
        roi = roi / self.depth_scale  # convert to meters

        ''' Method 2: valid_depths = valid_depths / self.depth_scale '''
        if not np.any(roi):
            return None, None, None, None, None, None, None, None

        # keep valid pixels within the sensor's usable range (depth_max drops
        # e.g. CARLA sky pixels at ~1000 m)
        valid = np.isfinite(roi) & (roi > 0) & (roi <= self.depth_max)
        rows, cols = np.where(valid)
        depths = roi[rows, cols]

        # find the z coordinate on the 3D BB
        if mask is not None:
            # Step 4/5: median depth of the mask is the object distance
            roi = depths
            if roi.size == 0:
                return None, None, None, None, None, None, None, None
            bb_center_z_coord = np.median(roi)
        else:
            roi = depths
            if roi.size == 0:
                return None, None, None, None, None, None, None, None
            bb_center_z_coord = (
                    depth_image[bbox_center_y][bbox_center_x] / self.depth_scale
            )

        # Step 6: keep only depths within the box-thickness window around the
        # object distance — mask bleed otherwise pulls background pixels in and
        # inflates the box to tens of meters along the view axis
        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= (self.depth_box_thickness / 2.0)
        if not np.any(mask_z):
            return None, None, None, None, None, None, None, None

        roi = roi[mask_z]
        z_min, z_max = np.min(roi), np.max(roi)
        z = float(bb_center_z_coord)

        if z == 0:
            return None, None, None, None, None, None, None, None

        # Step 7. Project from image to world space
        u, v = bbox_center_x, bbox_center_y  # xywh[0:2] or np.mean(mask_indices, axis=0)
        cx, cy = self.camera_models["depth"].cx(), self.camera_models["depth"].cy()
        fx, fy = self.camera_models["depth"].fx(), self.camera_models['depth'].fy()
        x = z * (u - cx) / fx
        y = z * (v - cy) / fy

        size_x = z * (bbox_size_x / fx)
        size_y = z * (bbox_size_y / fy)
        size_z = float(z_max - z_min)

        if len(rows) == 0:
            return None, None, None, None, None, None, None, None

        # Convert depth to 3D points
        xs = (cols - cx) * depths / fx
        ys = (rows - cy) * depths / fy

        points_3d = np.column_stack((xs, ys, depths))
        return x, y, z, size_x, size_y, size_z, None, points_3d

    def project_to_3d_with_pointcloud(self, mask, xywh, points=None):
        # todo: refactor to handle multiple detections
        # todo: use points and avoid using self.o3d_pointcloud
        # todo: refactor this to also use bbox instead of masks only
        if self.o3d_pointcloud.is_empty() and points is None:
            # Ensure the point cloud is in memory
            self.get_logger().info(f"Pointcloud is empty")
            return None, None, None, None, None, None, None, None

        bbox_center_x, bbox_center_y = map(int, xywh[:2])
        bbox_size_x, bbox_size_y = map(int, xywh[2:])
        u, v = bbox_center_x, bbox_center_y  # xywh[0:2] or np.mean(mask_indices, axis=0)
        cx, cy = self.camera_models["rgb"].cx(), self.camera_models["rgb"].cy()
        fx, fy = self.camera_models["rgb"].fx(), self.camera_models['rgb'].fy()

        mask_data = (mask.data[0, :, :] * 255)
        mask_data = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(mask_data)).to(o3c.Dtype.Int64).to(self.o3d_device)
        mask_xy = mask.xy[0]

        if points is None:
            points = self.o3d_pointcloud.point.positions.clone()

        # Step 1: Apply the extrinsics to the camera frame from the current frame, i.e apply camera -> self.output_frame extrinsics
        points_ = points.clone()

        if self.output_frame_to_rgb_tf_o3d is None or not self.static_camera_to_robot_tf:
            # source_frame = self.output_frame or self.frame_ids['pointcloud']
            source_frame = self.frame_ids['pointcloud']
            if self.output_frame and self.pointcloud_preprocessor.transform_pointcloud:
                source_frame = self.output_frame

            transform = self.lookup_transform(
                    source_frame, self.frame_ids['rgb'], rclpy.time.Time())

            if transform is not None:
                self.output_frame_to_rgb_tf = self.transform_to_matrix(transform)  # output_frame_to_rgb_tf
                self.output_frame_to_rgb_tf_o3d = o3c.Tensor(self.output_frame_to_rgb_tf, dtype=o3c.float32, device=self.o3d_device)

        if self.output_frame_to_rgb_tf_o3d is not None:
            points_ = self.o3d_pointcloud.clone().transform(self.output_frame_to_rgb_tf_o3d).point.positions  # points.transform(extrinsic_matrix_o3d)

        # # project back to the camera frame if the points were projected to the robots frame to project to the image
        # if self.output_frame and (self.camera_to_robot_tf_o3d is not None):
        #     # self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf_o3d.inv())
        #     points = self.o3d_pointcloud.point.positions_inv

        # Step 2: Project 3D points to 2D image plane
        x_ = points_[:, 0]
        y_ = points_[:, 1]
        z_ = points_[:, 2]
        x_2d = (x_ * fx / z_) + cx
        y_2d = (y_ * fy / z_) + cy

        # Step 3: remove points behind the camera and outside the RGB camera fov.
        # Bounds use the camera image size (the intrinsics' pixel space), NOT the
        # mask size — the mask is in the inference-image space (e.g. 640x640).
        image_width = self.camera_models['rgb'].width
        image_height = self.camera_models['rgb'].height
        valid_points_mask = (z_ > 0) & (x_2d >= 0) & (x_2d < image_width) & \
                       (y_2d >= 0) & (y_2d < image_height)  # Find points that project into the image

        # Filter valid projected points
        # valid_projected_points = self.o3d_pointcloud.point.positions[valid_points_mask]
        valid_projected_points = self.o3d_pointcloud.select_by_mask(valid_points_mask)  # points.select_by_mask(valid_points_mask)

        if valid_projected_points.is_empty():
            return None, None, None, None, None, None, None, None

        # scale camera-space pixel coords into the mask's (inference-image) space
        # before indexing, otherwise the lookup is misaligned and offset
        mask_height, mask_width = mask_data.shape[0], mask_data.shape[1]
        valid_x_2d = (x_2d[valid_points_mask] * (mask_width / image_width)).to(o3c.Dtype.Int64)
        valid_y_2d = (y_2d[valid_points_mask] * (mask_height / image_height)).to(o3c.Dtype.Int64)

        # Get mask values for these points. todo: could also use RGB image
        mask_values = mask_data[valid_y_2d, valid_x_2d]

        # Select points that fall within the mask. todo: could use bounding boxes, i.e in the rgb image, find points that land in bounding boxes
        # masked_points = valid_projected_points[mask_values > 0]
        masked_points = valid_projected_points.select_by_mask(mask_values > 0)

        # Step 4: cluster points and get the bounding box. todo: move this outside of the loop to perform once
        clusters, bboxes, centers, extents, quats = self.cluster_points(masked_points)
        # only select the cluster with the largest label
        # get the center (if not using the bbox)
        # centroid = np.mean(clustered_points, axis=0)
        if bboxes:
            x, y, z = centers[0]
            size_x, size_y, size_z = extents[0]
            return x, y, z, size_x, size_y, size_z, quats[0], clusters
        return None, None, None, None, None, None, None, None

    def cluster_points(self, o3d_pcd):
        """
        Perform DBSCAN clustering and return clusters with their bounding boxes.
        Return the list of clusters.
        Todo: call the "get_clusters" method in euclidean_clustering node
        """
        # self.get_logger().info("Clustering point cloud with DBSCAN...")

        # Ensure the point cloud is in memory
        if o3d_pcd.is_empty():
            self.get_logger().info(f"Pointcloud is empty")
            return [], [], [], [], []

        # Use GPU DBSCAN clustering
        labels = o3d_pcd.cluster_dbscan(
            eps=self.cluster_tolerance,
            min_points=self.min_cluster_size,
            print_progress=True  # todo: set to False
        )
        # o3d_pcd.point.labels = labels  # if adding label, create a new Pointcloud Tensor Geometry object in each callback

        # Get unique labels (excluding noise points labeled as -1)
        labels = torch.utils.dlpack.from_dlpack(labels.to_dlpack())
        # labels = labels.cpu().numpy()
        # unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)  # largest_cluster_label = max(labels, key=lambda l: np.sum(labels == l))
        unique_labels, counts = torch.unique(labels[labels != -1], return_counts=True)
        # unique_labels, counts = set(labels.tolist()), None

        if len(unique_labels) == 0:
            self.get_logger().info(f"unique labels: {unique_labels}, len: {len(unique_labels)}")
            return  [], [], [], [], []  # Return empty lists if no valid clusters

        # unique_labels = unique_labels[unique_labels >= 0]

        max_label = labels.max().item()
        self.get_logger().info(f"DBSCAN found {max_label + 1} clusters")

        # if self.max_cluster_size > 0 and counts is not None:
        #     counts_less_than_cluster_size = counts < self.max_cluster_size
        #     unique_labels = unique_labels[counts_less_than_cluster_size]
        #     # counts = counts[counts_less_than_cluster_size]

        clusters = []
        bboxes = []
        centers, extents, quats = [], [], []

        for label in unique_labels:
            # Create mask for current cluster
            mask = (labels == label)

            # we convert the boolean array to an integer array since dlpack does not support zero-copy transfer for bool
            mask = mask.to(device=self.torch_device, dtype=torch.uint8)
            mask = o3c.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(mask))  # o3c.Tensor(mask, device=self.o3d_device)
            mask = mask.to(o3c.Dtype.Bool)  # convert back to a boolean mask

            # Create new pointcloud for cluster
            cluster_pcd = o3d_pcd.select_by_mask(mask)

            if cluster_pcd.is_empty():
                continue

            points = cluster_pcd.point.positions

            # # remove large clusters. This is redundant since we already removed unique_labels with counts < self.max_cluster_size
            # if 0 < self.max_cluster_size < points.shape[0]:
            #     continue

            # Get cluster height
            min_z = points[:, 2].min().item()
            max_z = points[:, 2].max().item()
            height = max_z - min_z

            # # Filter clusters by height. height < self.cluster_min_height or height > self.cluster_max_height
            # if not (self.cluster_min_height <= height <= self.cluster_max_height):
            #     continue

            clusters.append(cluster_pcd)

            if self.bounding_box_type.lower() == "aabb":
                bounding_box = cluster_pcd.get_axis_aligned_bounding_box()
                center = bounding_box.get_center().cpu().numpy().tolist()
                extent = bounding_box.get_extent().cpu().numpy().tolist()
                quat = None  # aabb does not have orientation
            elif self.bounding_box_type.lower() == "obb":
                bounding_box = cluster_pcd.get_oriented_bounding_box()
                center = bounding_box.center.cpu().numpy().tolist()
                extent = bounding_box.extent.cpu().numpy().tolist()
                # Convert rotation matrix to quaternion
                R = bounding_box.rotation.cpu().numpy()
                quat = quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))
                quat = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))
            else:
                raise ValueError(f"Unknown bounding box type: {self.bounding_box_type}")

            bboxes.append(bounding_box)
            centers.append(center)
            extents.append(extent)
            quats.append(quat)

        return clusters, bboxes, centers, extents, quats

    def create_3d_detection(self, x, y, z, size_x, size_y, size_z, confidence, class_id, quat=None, frame_id=None):
        det_msg = Detection3D()
        # frame_id: 3D detections live in output_frame (base_link by default)
        if frame_id:
            det_msg.header.frame_id = frame_id
        det_msg.bbox.center.position.x = float(x)
        det_msg.bbox.center.position.y = float(y)
        det_msg.bbox.center.position.z = float(z)
        if quat is not None:
            det_msg.bbox.center.orientation = quat
        det_msg.bbox.size.x = float(size_x)  # Width
        det_msg.bbox.size.y = float(size_y)  # Height
        det_msg.bbox.size.z = float(size_z)  # Depth approximation

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_id
        hypothesis.hypothesis.score = float(confidence)
        det_msg.results.append(hypothesis)
        return det_msg

    def create_marker(self, marker_id, x, y, z, size_x, size_y, size_z,
                      frame_id, timestamp=None, confidence=None, class_id=None, track_id=None, quat=None,
                      rgba=None):
        if rgba is None:
            rgba = [1.0, 0.0, 0.0, 0.5]
        marker = Marker()
        marker.header.frame_id = frame_id
        if timestamp is None:
            timestamp = self.get_clock().now().to_msg()
        marker.header.stamp = timestamp
        marker.ns = "image_obstacles"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        if quat is not None:
            marker.pose.orientation = quat
        marker.scale.x = size_x
        marker.scale.y = size_y
        marker.scale.z = size_z

        # Set color (using track_id to generate unique colors). for now doesn't work
        if track_id is not None:
            color_hash = hash(str(track_id))
            marker.color.r = float((color_hash & 0xFF0000) >> 16) / 255.0
            marker.color.g = float((color_hash & 0x00FF00) >> 8) / 255.0
            marker.color.b = float(color_hash & 0x0000FF) / 255.0
        else:
            marker.color.r = rgba[0]
            marker.color.g = rgba[1]
            marker.color.b = rgba[2]
            marker.color.a = rgba[3]

        marker.lifetime = rclpy.duration.Duration(seconds=0.5).to_msg()  # 0.1 todo: set as a parameter
        return marker

    def lookup_transform(self, source_frame_id, target_frame_id, timestamp=None):
        if timestamp is None:
            timestamp = rclpy.time.Time()

        # Try to get the transform from camera to robot
        try:
            transform = self.tf_buffer.lookup_transform(
                    target_frame_id,
                    source_frame_id,
                    # this could also be the depth msg timestamp. use "rclpy.time.Time()" to get the latest
                    timestamp,
                    rclpy.duration.Duration(seconds=self.transform_timeout)
            )
        except tf2_ros.LookupException as e:
            self.get_logger().error(f"TF Lookup Error: {str(e)}")
            return None
        except tf2_ros.ConnectivityException as e:
            self.get_logger().error(f"TF Connectivity Error: {str(e)}")
            return None
        except tf2_ros.ExtrapolationException as e:
            self.get_logger().error(f"TF Extrapolation Error: {str(e)}")
            return None
        return transform

    def get_camera_to_robot_tf(self, source_frame_id, timestamp=None):
        if self.camera_to_robot_tf is not None and self.static_camera_to_robot_tf:
            return

        if timestamp is None:
            timestamp = rclpy.time.Time()
        if self.output_frame:
            transform = self.lookup_transform(source_frame_id, self.output_frame, timestamp)

            # Convert the TF transform to a 4x4 transformation matrix
            if transform is not None:
                self.camera_to_robot_tf = self.transform_to_matrix(transform)
                if self.use_pointcloud:
                    self.camera_to_robot_tf_o3d = o3c.Tensor(self.camera_to_robot_tf,
                                                             dtype=o3c.float32, device=self.o3d_device)
            return

    def transform_to_matrix(self, transform: TransformStamped):
        """Convert TransformStamped to 4x4 transformation matrix."""
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        matrix = quaternion_matrix([rotation.x, rotation.y, rotation.z, rotation.w])
        matrix[:3, 3] = [translation.x, translation.y, translation.z]

        # tf_matrix = o3c.Tensor(matrix, dtype=o3c.float32, device=self.o3d_device)
        # self.camera_to_robot_tf = tf_matrix
        return matrix

    @staticmethod
    def transform_box_to_frame(x, y, z, size_x, size_y, size_z, tf_matrix):
        """Transform an axis-aligned box with a 4x4 matrix: the center gets the full
        transform, the size only the rotation (|R|, no translation)."""
        center = tf_matrix @ np.array([x, y, z, 1.0])
        size = np.abs(tf_matrix[:3, :3]) @ np.array([size_x, size_y, size_z])
        return (float(center[0]), float(center[1]), float(center[2]),
                float(size[0]), float(size[1]), float(size[2]))

    def transform_bbox_3d(self, x, y, z, size_x, size_y, size_z, points_3d):
        # transform the pose
        object_pose_camera_frame = np.array([x, y, z, 1])
        object_pose_robot_frame = np.dot(self.camera_to_robot_tf, object_pose_camera_frame)
        x_robot, y_robot, z_robot = object_pose_robot_frame[:3]

        # transform the size
        object_size_camera_frame = np.array([size_x, size_y, size_z, 1])
        object_size_robot_frame = np.dot(self.camera_to_robot_tf, object_size_camera_frame)
        size_x_robot, size_y_robot, size_z_robot = object_size_robot_frame[:3]

        # transform the points
        points_3d_homogenous_camera_frame = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        points_3d_homogenous_robot_frame = np.dot(self.camera_to_robot_tf, points_3d_homogenous_camera_frame.T).T
        points_3d_robot = points_3d_homogenous_robot_frame[:, :3]
        return x_robot, y_robot, z_robot, size_x_robot, size_y_robot, size_z_robot, points_3d_robot

    def convert_to_open3d_tensor(self, input_array):
        if isinstance(input_array, np.ndarray):
            # could also initialize as an Open3D tensor directly
            input_array = o3c.Tensor(input_array, device=self.o3d_device)

        if isinstance(input_array, torch.Tensor):
            input_array = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(input_array)).to(device=self.o3d_device)

        return input_array

    def normalize_depth_image(self, depth_image, max_val=255, dtype=None):
        """

        :param depth_image:
        :param max_val: Either 1 or 255
        :param dtype: Either cv2.CV_32F or cv2.CV_8UC1
        :return:
        """
        if self.use_gpu or (not self.use_gpu):
            if dtype is None:
                dtype = torch.uint8
                if max_val == 1:
                    dtype = torch.float32

            if isinstance(depth_image, np.ndarray):
                # could also initialize as an Open3D tensor directly
                depth_image = torch.from_numpy(depth_image).to(device=self.torch_device, dtype=torch.float32)

            # Normalize the depth image to fall between 0 and max_val
            depth_image_normalized = (depth_image - depth_image.min()) / (
                        depth_image.max() - depth_image.min()) * max_val
            depth_image_normalized = depth_image_normalized.to(device=self.torch_device, dtype=dtype)
            return depth_image_normalized

        if dtype is None:
            dtype = cv2.CV_8UC1
            if max_val == 1:
                dtype = cv2.CV_32F

        # note, if the depth is a single file and Opencv<4.7.0, then it has been normalized
        # We need to restore
        depth_image_normalized = cv2.normalize(
                depth_image, depth_image, 0, max_val, cv2.NORM_MINMAX,
                dtype=dtype)  # 1 or 255 depends on datatype, 1 for float, e.g 32F and 255 for int eg 8U

        return depth_image_normalized

    def smooth_mask_and_extract_moment(self, mask_xy):
        smoothed_masks = [cv2.approxPolyDP(mask, 4, True) for mask in mask_xy]

        for smoothed_mask in smoothed_masks:
            M = cv2.moments(smoothed_mask)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])

            scale_ratio = 1.1
            resized_mask = smoothed_mask.copy()

            for p in resized_mask:
                p[0][0] = (p[0][0] - cx) * scale_ratio + cx
                p[0][1] = (p[0][1] - cy) * scale_ratio + cy

            xy = [(p[0][0], p[0][1]) for p in resized_mask]

    def parameter_change_callback(self, params):
        """
        Todo:
            * change topics (input/output) and destroy subscribers/publishers
        Triggered whenever there is a change request for one or more parameters.

        Args:
            params (List[Parameter]): A list of Parameter objects representing the parameters that are
                being attempted to change.

        Returns:
            SetParametersResult: Object indicating whether the change was successful.
        """
        result = SetParametersResult()
        result.successful = True

        # Iterate over each parameter in this node
        for param in params:
            if param.name == 'publish_object_array' and param.type_ == Parameter.Type.BOOL:
                self.publish_object_array = param.value
                if self.publish_object_array and not hasattr(self, 'object_array_pub'):
                    self.object_array_pub = self.create_publisher(ObjectArray, 'yolo/objects', self.queue_size)
            elif param.name == 'publish_obstacle_array' and param.type_ == Parameter.Type.BOOL:
                self.publish_obstacle_array = param.value
                if self.publish_obstacle_array:
                    if NAV2_DYNAMIC_MSGS_AVAILABLE:
                        if not hasattr(self, 'obstacle_detection_pub'):
                            self.obstacle_detection_pub = self.create_publisher(ObstacleArray, 'yolo/obstacles', self.queue_size)
                    else:
                        if not getattr(self, '_nav2_warned', False):
                            self.get_logger().warning("nav2_dynamic_msgs not available; nav2 ObstacleArray output disabled")
                            self._nav2_warned = True
            elif param.name == 'publish_debug_image' and param.type_ == Parameter.Type.BOOL:
                self.publish_debug_image = param.value
            elif param.name == 'model_path' and param.type_ == Parameter.Type.STRING:
                self.model_path = param.value
                # todo: load the model
            elif param.name == 'track_2d' and param.type_ == Parameter.Type.BOOL:
                self.track_2d = param.value
            elif param.name == 'tracker_2d.path' and param.type_ == Parameter.Type.STRING:
                self.tracker_2d_cfg['path'] = param.value
            # todo: add other tracker params
            elif param.name == 'plot_tracks' and param.type_ == Parameter.Type.BOOL:
                self.plot_tracks = param.value
            elif param.name == 'use_gpu' and param.type_ == Parameter.Type.BOOL:
                self.use_gpu = False

                # Then check GPU availability if use_gpu
                use_gpu = param.value
                self.device = 'cpu'
                self.torch_device = torch.device('cpu')
                if use_gpu:
                    if torch.cuda.is_available():
                        self.device = 'cuda:0'
                        self.torch_device = torch.device('cuda:0')
                        self.use_gpu = True
                    else:
                        self.use_gpu = False
                        result.successful = False
                        result.reason = "Torch was not installed/built with CUDA support. GPU backend cannot use torch functions."
                        self.get_logger().warn("Torch was not installed/built with CUDA support. "
                                               "GPU backend cannot use torch functions.")
                self.inference_dict['device'] = self.device
            elif param.name == 'show_image' and param.type_ == Parameter.Type.BOOL:
                self.show_image = param.value
                if not self.show_image:
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
            elif param.name == 'use_image_dimensions' and param.type_ == Parameter.Type.BOOL:
                self.use_image_dimensions = param.value
            elif param.name == 'image_dimensions' and param.type_ == Parameter.Type.INTEGER_ARRAY:
                self.image_dimensions = param.value
            elif param.name == 'resize_image' and param.type_ == Parameter.Type.BOOL:
                self.resize_image = param.value
            elif param.name == 'half_precision' and param.type_ == Parameter.Type.BOOL:
                self.half_precision = param.value
                self.inference_dict['half'] = self.half_precision
            elif param.name == 'conf_thresh' and param.type_ == Parameter.Type.DOUBLE:
                self.conf_thresh = param.value
                self.inference_dict['conf'] = self.conf_thresh
            elif param.name == 'iou_thresh' and param.type_ == Parameter.Type.DOUBLE:
                self.iou_thresh = param.value
                self.inference_dict['iou'] = self.iou_thresh
            elif param.name == 'max_det' and param.type_ == Parameter.Type.INTEGER:
                self.max_det = param.value
                self.inference_dict['max_det'] = self.max_det
            elif param.name == 'classes' and param.type_ in (Parameter.Type.STRING_ARRAY, Parameter.Type.INTEGER_ARRAY):
                classes = param.value
                if param.type_ == Parameter.Type.STRING_ARRAY:
                    assert all(x in self.supported_class_names for x in classes)
                    self.classes = [self.class_names_inv[x.strip()] for x in classes]
                else:
                    assert all(x in self.supported_class_keys for x in classes)
                    self.classes = classes
                self.inference_dict['classes'] = self.classes
            elif param.name == 'update_class' and param.type_ == Parameter.Type.STRING:
                # Update the list of classes based on the update_class parameter.
                # For CLI, add -- before -class,
                # e.g ros2 param set /single_stream_detector update_class -- -truck.
                self.update_class = param.value
                mode = "add"
                cls = self.update_class
                if self.update_class.startswith('-'):
                    mode = "remove"
                    cls = self.update_class[1:]
                # check if the class name is supported
                if cls not in self.supported_class_names:
                    result.successful = False
                    result.reason = f"'{cls}' is not a supported class name."
                    self.get_logger().warn(f"'{cls}' is not a supported class name.")
                # get the class key
                cls_key = self.class_names_inv.get(cls.strip(), False)
                # add or remove the class
                if (mode == "add") and (cls_key not in self.classes):
                    self.classes.append(cls_key)
                    print(f"Added '{cls}' to the list of classes.")
                elif (mode == "remove") and cls_key:
                    self.classes.remove(cls_key)
                    print(f"Removed '{cls}' from the list of classes.")
                self.inference_dict['classes'] = self.classes
                # update the classes parameter
                self.set_parameters(
                        [
                            rclpy.parameter.Parameter(
                                'classes',
                                Parameter.Type.STRING_ARRAY,
                                [self.class_names[x] for x in self.classes]
                            )
                        ]
                )
            elif param.name == 'agnostic_nms' and param.type_ == Parameter.Type.BOOL:
                self.agnostic_nms = param.value
                self.inference_dict['agnostic_nms'] = self.agnostic_nms
            elif param.name == 'augment' and param.type_ == Parameter.Type.BOOL:
                self.augment = param.value
                self.inference_dict['augment'] = self.augment
            elif param.name == 'verbose' and param.type_ == Parameter.Type.BOOL:
                self.verbose = param.value
                os.environ['YOLO_VERBOSE'] = str(self.verbose)
                self.inference_dict['verbose'] = self.verbose
            elif param.name == 'static_camera_info' and param.type_ == Parameter.Type.BOOL:
                self.static_camera_info = param.value
            elif param.name == 'publish_empty_detections' and param.type_ == Parameter.Type.BOOL:
                self.publish_empty_detections = param.value
            elif param.name == 'depth_box_thickness' and param.type_ == Parameter.Type.DOUBLE:
                self.depth_box_thickness = param.value
            elif apply_profiler_param(self.profiler, param.name, param.value):
                pass
            else:
                result.successful = False
            self.get_logger().info(f"Success = {result.successful} for param {param.name} to value {param.value}")
        return result

    def destroy_node(self):
        # close OpenCV windows
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        # the reference to the model
        del self.model
        # clear the cuda cache
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()
        # close the temporary file used for the custom tracker settings
        if self.track_2d:
            try:
                self.temp_file.close()
            except FileNotFoundError:
                pass
        # delete the open3d pointlcoud object cleanly
        if self.project_to_3d and self.use_pointcloud:
            self.pointcloud_preprocessor.o3d_pointcloud.clear()
            self.o3d_pointcloud.clear()
            del self.o3d_pointcloud, self.pointcloud_preprocessor.o3d_pointcloud
        return None


def main(args=None):
    rclpy.init(args=args)
    node = ImageObstacleDetectionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
