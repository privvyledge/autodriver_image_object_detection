"""
Single stream detector for object detection with optional tracking.
Usage:
    sudo apt-get install ros-${ROS_DISTRO}-vision-msgs
    ros2 run autodriver_image_object_detection single_stream_detector

Todo:
     1. Add verbose=False parameter to predict function [done]
     2. Setup parameter change callback [done]
     3. Add support for masking an image with an ROI, running inference then displaying the full image detection results. See:
        * https://github.com/ultralytics/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py
        * https://github.com/ultralytics/ultralytics/blob/main/ultralytics/solutions/trackzone.py
        * https://docs.ultralytics.com/guides/region-counting/#real-world-applications
     4. Add support for disabling plotting of masks, labels, boxes, probs, etc in show/publish_debug_image namespace [done: no need]
     5. Add add_class and remove_class parameters/services [done]
     6. Add support for snapshot mode. I.e triggers a service if num_detections > 0 for rosbag/video recording e.g recording motion only
"""
import os
import sys
import time
import uuid
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
except ImportError:
    print("nav2_dynamic_msgs not found. ")
import tf2_ros
from tf2_ros import TransformBroadcaster, TransformListener, Buffer, LookupException, ConnectivityException, \
    ExtrapolationException

from autodriver_image_object_detection.utils.common import pack_2d_detection, pack_nav2_obstacle_msg, pack_derived_object_msg, update_tracker_param
from autodriver_image_object_detection.utils.imaging_utils import parse_image_message

class SingleStreamDetector(Node):
    def __init__(self):
        this_package_dir = get_package_share_directory('autodriver_image_object_detection')
        super(SingleStreamDetector, self).__init__("single_stream_detector")

        # Declare parameters
        self.declare_parameter(name='input_image_topic', value="carla/ego_vehicle/rgb_front/image",  # "camera/image_raw", camera/color/image_raw, carla/ego_vehicle/rgb_front/image
                               descriptor=ParameterDescriptor(
                                   description='The input image topic. '
                                               'Works with all image types: RGB(A), BGR(A), mono8, mono16.',
                                   type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='input_camera_info_topic', value="carla/ego_vehicle/rgb_front/camera_info",  # camera/color/camera_info, carla/ego_vehicle/rgb_front/camera_info
                               # "camera/camera_info",
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

        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
            description='',
            type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='model_path', value="yolo11n-seg.engine",
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
        self.declare_parameter("classes", ['person', 'car', 'bicycle', 'motorcycle', 'bus',
                                           'truck'])  # [] or ['person', 'car'] or [0, 2]
        self.declare_parameter("update_class", "")  # type "class_name" to add or "-class_name" to delete
        self.declare_parameter("agnostic_nms", True)
        self.declare_parameter("augment", False)
        self.declare_parameter("verbose", False)
        self.declare_parameter('static_camera_info', True)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.input_camera_info_topic = self.get_parameter('input_camera_info_topic').get_parameter_value().string_value
        self.input_image_topic_is_compressed = self.get_parameter(
            'input_image_topic_is_compressed').get_parameter_value().bool_value
        self.detection_results_topic = self.get_parameter('detection_results_topic').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.detection_image_topic = self.get_parameter('detection_image_topic').get_parameter_value().string_value
        self.segmentation_image_topic = self.get_parameter(
            'segmentation_image_topic').get_parameter_value().string_value
        self.segmentation_mask_image_topic = self.get_parameter(
            'segmentation_mask_image_topic').get_parameter_value().string_value
        self.qos = self.get_parameter('qos').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.export_model_format = self.get_parameter('export_model_format').get_parameter_value().string_value
        self.track_2d = self.get_parameter('track_2d').get_parameter_value().bool_value
        self.tracker_2d_cfg = self.get_parameters_by_prefix('tracker_2d')
        self.tracker_2d_cfg = {k: v.value for k, v in self.tracker_2d_cfg.items()}
        self.plot_tracks = self.get_parameter('plot_tracks').get_parameter_value().bool_value
        self.queue_size = self.get_parameter('queue_size').get_parameter_value().integer_value
        self.use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.use_image_dimensions = self.get_parameter("use_image_dimensions").get_parameter_value().bool_value
        self.image_dimensions = self.get_parameter("image_dimensions").get_parameter_value().integer_array_value
        self.resize_image = self.get_parameter("resize_image").get_parameter_value().bool_value
        self.half_precision = self.get_parameter("half_precision").get_parameter_value().bool_value
        self.conf_thresh = self.get_parameter("conf_thresh").get_parameter_value().double_value
        self.iou_thresh = self.get_parameter("iou_thresh").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.classes = self.get_parameter("classes").value
        self.update_class = self.get_parameter("update_class").value
        self.agnostic_nms = self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.verbose = self.get_parameter("verbose").get_parameter_value().bool_value
        self.static_camera_info = self.get_parameter('static_camera_info').get_parameter_value().bool_value

        os.environ['YOLO_VERBOSE'] = str(self.verbose)

        # Setup the device
        self.device = 'cpu'
        self.torch_device = torch.device('cpu')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'  # 'cuda'
                self.torch_device = torch.device('cuda:0')
            else:
                self.use_gpu = False

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
                # self.model_path = self.model_path.split('.')[0] + '.' + self.model_path.split('.')[-1]

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
        self.camera_info = None
        self.camera_model = PinholeCameraModel()

        try:
            self.get_logger().info("Fusing model...")
            self.model.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fusing the model: {e}. "
                                   f"This usually occurs if not using a pytorch model (.pt), "
                                   f"e.g a TensorRT model (.engine)")

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

        # Setup dynamic parameter reconfiguring.
        # Register a callback function that will be called whenever there is an attempt to
        # change one or more parameters of the node.
        self.add_on_set_parameters_callback(self.parameter_change_callback)

        # Subscribers
        self.image_sub = self.create_subscription(
                self.image_message_type,
                self.input_image_topic,
                self.image_callback,
                qos_profile)
        self.camera_info_sub = self.create_subscription(
                CameraInfo,
                self.input_camera_info_topic,
                self.camera_info_callback,
                qos_profile)

        # Publishers
        self.detection_results_pub = self.create_publisher(
                Detection2DArray,
                self.detection_results_topic,
                self.queue_size,  # qos_profile
        )

        self.object_array_pub = self.create_publisher(
                ObjectArray,
                'yolo/objects',
                self.queue_size,  # qos_profile
        )

        try:
            self.obstacle_detection_pub = self.create_publisher(ObstacleArray, 'yolo/obstacles', qos_profile)
        except NameError:
            pass

        if self.publish_debug_image:
            self.detection_image_pub = self.create_publisher(
                    self.image_message_type,
                    self.detection_image_topic,
                    self.queue_size  # qos_profile,  # use Best Effort for publishing
            )
            self.segmentation_image_pub = self.create_publisher(
                    self.image_message_type,
                    self.segmentation_image_topic,
                    self.queue_size  # qos_profile,  # use Best Effort for publishing
            )
            self.segmentation_mask_image_pub = self.create_publisher(
                    self.image_message_type,
                    self.segmentation_mask_image_topic,
                    self.queue_size  # qos_profile,  # use Best Effort for publishing
            )

        self.get_logger().info(
            (
                f"single_stream_detector started. "
                f"Publishing on {self.detection_results_topic}. "
                f"Subscribing to {self.input_image_topic}."
            )
        )

    def image_callback(self, msg):
        if self.camera_info is None:
            self.get_logger().warn("No CameraInfo received yet — skipping frame.", once=True)
            return
        try:
            msg_timestamp = None
            msg_fmt = "bgr8"
            conversion = None
            inverse_conversion = None
            is_color = True
            is_depth = False
            cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt, conversion, inverse_conversion, is_color, is_depth, compressed_msg_codec = parse_image_message(
                msg, self.bridge, self.image_message_format, logger=self.get_logger())

            # get image dimensions
            if self.imgsz is None:
                self.image_height, self.image_width = cv_image.shape[:2]
                if self.use_image_dimensions:
                    # Check divisibility by 32 to conform to YOLOs convolution kernel size and stride length
                    new_height = self.image_height if self.image_height % 32 == 0 else ((
                                                                                                    self.image_height // 32) + 1) * 32
                    new_width = self.image_width if self.image_width % 32 == 0 else ((self.image_width // 32) + 1) * 32

                    self.imgsz = (new_height, new_width)
                    self.get_logger().info(f"Using image dimensions: {self.imgsz}")
                else:
                    self.imgsz = list(self.image_dimensions)  # [640, 640]

            self.inference_dict['imgsz'] = self.imgsz
            # (optional) resize the image
            if self.resize_image and self.use_image_dimensions and (
                    (self.image_height, self.image_width) != (self.imgsz[0], self.imgsz[1])):
                cv_image = cv2.resize(cv_image, (self.imgsz[1], self.imgsz[0]))  # , interpolation=cv2.INTER_LINEAR

            # detect/track objects in the visual image
            self.detect_objects(cv_image)
            detection_msg, detection_image, mask_img = self.parse_results(self.results, msg.header)

            # publish the detection array results
            self.detection_results_pub.publish(detection_msg)

            # convert OpenCV image back to the input msg_fmt
            if detection_image is not None:
                if self.publish_debug_image:
                    if conversion is not None:
                        detection_image = cv2.cvtColor(detection_image, inverse_conversion)

                    if self.image_message_format in ("compressed", "packet"):
                        detection_image_msg = self.bridge.cv2_to_compressed_imgmsg(detection_image,
                                                                              dst_format=compressed_msg_codec)  # msg.format.split(';')[1].split()[0]
                    else:
                        detection_image_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding=msg_fmt)

                    detection_image_msg.header.frame_id = image_frame_id
                    detection_image_msg.header.stamp = msg_timestamp

                    if self.detection_image_topic:
                        self.detection_image_pub.publish(detection_image_msg)

                    if self.segmentation_mask_image_topic and (mask_img is not None):
                        if self.image_message_format in ("compressed", "packet"):
                            mask_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                                    mask_img,
                                    dst_format=compressed_msg_codec)  # msg.format.split(';')[1].split()[0]
                        else:
                            mask_image_msg = self.bridge.cv2_to_imgmsg(
                                    mask_img,
                                    encoding="mono8")

                        mask_image_msg.header.frame_id = image_frame_id
                        mask_image_msg.header.stamp = msg_timestamp
                        self.segmentation_mask_image_pub.publish(mask_image_msg)

                    if self.segmentation_image_topic and (mask_img is not None):
                        cv_image_inverted = cv_image
                        if inverse_conversion is not None:
                            cv_image_inverted = cv2.cvtColor(cv_image, inverse_conversion)
                        color_mask_img = cv2.bitwise_and(cv_image_inverted, cv_image_inverted, mask=mask_img)
                        if self.show_image:
                            cv2.imshow("color_mask_image", color_mask_img)
                            cv2.waitKey(1)

                        if self.image_message_format in ("compressed", "packet"):
                            color_mask_image_msg = self.bridge.cv2_to_compressed_imgmsg(
                                    color_mask_img,
                                    dst_format=compressed_msg_codec)  # msg.format.split(';')[1].split()[0]
                        else:
                            color_mask_image_msg = self.bridge.cv2_to_imgmsg(
                                    color_mask_img,
                                    encoding=msg_fmt)  # passthrough

                        color_mask_image_msg.header.frame_id = image_frame_id
                        color_mask_image_msg.header.stamp = msg_timestamp
                        self.segmentation_image_pub.publish(color_mask_image_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            # if self.debug:
            #     raise e


    def camera_info_callback(self, msg):
        if self.camera_info is None:
            self.camera_info = msg
            self.camera_model.fromCameraInfo(msg)

        # update the camera infos and models if not static
        if not self.static_camera_info:
            self.camera_info = msg
            self.camera_model.fromCameraInfo(msg)

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
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = header.stamp  # self.get_clock().now().to_msg()
        detections_msg.header.frame_id = header.frame_id

        objects_msg = ObjectArray()
        objects_msg.header.stamp = header.stamp  # self.get_clock().now().to_msg()
        objects_msg.header.frame_id = header.frame_id

        try:
            obstacle_msg = ObstacleArray()
            obstacle_msg.header.stamp = header.stamp  # self.get_clock().now().to_msg()
            obstacle_msg.header.frame_id = header.frame_id
        except NameError as e:
            self.get_logger().error(f'ObstacleArray message not found.')
            obstacle_msg = None

        mask_img = None

        for result in results:
            self.detection_image = result.plot(
                    conf=True,
                    labels=True,
                    boxes=True,
                    masks=True,
                    probs=True,
                    # # todo: use the image below to specify the original image if passing an ROI masked image to the detector
                    # img=self.inference_dict['source'],  # numpy image to overlay detections on. This is slower since it needs to be transferred to GPU
                    # im_gpu=None,  # torch tensor image to overlay detections on. This is faster since it does not need to be transferred to GPU
            )
            if self.show_image:
                # Visualize the results on the frame
                cv2.imshow("image", self.detection_image)
                cv2.waitKey(1)

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
                    cv2.imshow("masked_image", mask_img)
                    cv2.waitKey(1)
                
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
                        bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)), conf,
                        id=track_ids[i] if track_ids is not None else -1, z_size=0.0)
                    obstacle_msg.obstacles.append(obstacle_2d)

            self.object_array_pub.publish(objects_msg)

            if obstacle_msg is not None:
                self.obstacle_detection_pub.publish(obstacle_msg)

            return detections_msg, mask_img
        return None

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
            if param.name == 'publish_debug_image' and param.type_ == Parameter.Type.BOOL:
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
                    cv2.destroyAllWindows()
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
            else:
                result.successful = False
            self.get_logger().info(f"Success = {result.successful} for param {param.name} to value {param.value}")
        return result

    def destroy_node(self):
        # close OpenCV windows
        cv2.destroyAllWindows()
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


def main(args=None):
    rclpy.init(args=args)
    node = SingleStreamDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()