"""
Multi stream detector for object detection with optional tracking.
Supports inference from e.g multiple cameras, videos, rtsp streams, etc. using batch inference
Usage:
    sudo apt-get install ros-${ROS_DISTRO}-vision-msgs
    ros2 run autodriver_image_object_detection single_stream_detector

Todo:
    * ultralytics now supports tracking for multi-streams. However, stream=True must be passed.
    * add support for publishing ObstacleArray and ObjectArray messages similar to single_stream_detector and yolo_detection_node
"""

import time
import uuid
import struct
from pathlib import Path
from collections import defaultdict
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy import qos
from rclpy.time import Time
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
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
import tf_transformations
import numpy as np
import transforms3d
from tf_transformations import quaternion_matrix, quaternion_from_matrix
from cv_bridge import CvBridge
import cv2
import torch
import torch.utils.dlpack
import ultralytics
from autodriver_image_object_detection.utils.imaging_utils import parse_image_message


def pack_2d_detection(x, y, size_x, size_y, class_id, conf, id):
    detection = Detection2D()
    detection.bbox.center.position.x = float(x)
    detection.bbox.center.position.y = float(y)
    detection.bbox.size_x = float(size_x)
    detection.bbox.size_y = float(size_y)
    detection.id = str(id)
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_id
    hypothesis.hypothesis.score = float(conf)
    detection.results.append(hypothesis)
    return detection


def pack_nav2_obstacle_msg(x, y, size_x, size_y, class_id, conf, id=None):
    if id in (None, -1):
        uuid_ = uuid.uuid4()
    else:
        uuid_ = uuid.UUID(int=id)
        # or
        #id_str = str(id)
        #uuid_ = uuid.uuid5(uuid.NAMESPACE_DNS, id_str)

    obstacle_msg = Obstacle()
    obstacle_msg.uuid.uuid = list(uuid_.bytes)
    obstacle_msg.score = float(conf)
    obstacle_msg.position.x = float(x)
    obstacle_msg.position.y = float(y)
    obstacle_msg.size.x = float(size_x)
    obstacle_msg.size.y = float(size_y)
    return obstacle_msg


class MultiStreamDetector(Node):
    """docstring for ClassName"""
    def __init__(self):
        """Constructor for MultiStreamDetector"""
        super(MultiStreamDetector, self).__init__("multi_stream_detector")

        # Declare parameters
        self.declare_parameter('num_cameras', 2)
        self.declare_parameter(name='synchronization_interval', value=0.1,
                               descriptor=ParameterDescriptor(
                                       description='value < 0.0 disables synchronization meaning multiple subscriber callbacks will be created for each topic, '
                                                   'value == 0.0 uses exact synchronization which assumes the messages have the same timestamp, and '
                                                   'value > 0.0 uses approximate synchronization which assumes the messages have timestamps close to each other.',
                                       type=ParameterType.PARAMETER_DOUBLE
        ))
        # self.declare_parameter(name='input_image_topics', value=["/camera/camera0/color/image_raw",
        #                                                         "/camera/camera1/color/image_raw"],
        #                        descriptor=ParameterDescriptor(
        #                                description='The input image topics. '
        #                                            'Works with all image types: RGB(A), BGR(A), mono8, mono16.',
        #                                type=ParameterType.PARAMETER_STRING_ARRAY
        #                        ))
        #
        # self.declare_parameter(name='input_camera_info_topics', value=["/camera/camera0/color/camera_info",
        #                                                               "/camera/camera1/color/camera_info"],
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING_ARRAY
        #                        ))
        self.declare_parameter(name='input_image_topic_is_compressed', value=[False, False],
                               descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL_ARRAY
        ))
        # self.declare_parameter(name='detection_results_topic', value=["/yolo/detection_results0",
        #                                                               "/yolo/detection_results1"],
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING_ARRAY
        #                        ))
        self.declare_parameter('publish_debug_image', True)
        # self.declare_parameter(name='detection_image_topic', value=["/yolo/detection_image0",
        #                                                             "/yolo/detection_image1"],
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING_ARRAY
        #                        ))
        # self.declare_parameter(name='segmentation_image_topic', value=["/yolo/segmentation_image0",
        #                                                               "/yolo/segmentation_image1"],
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING_ARRAY
        #                        ))
        # self.declare_parameter(name='segmentation_mask_image_topic', value=["/yolo/segmentation_mask_image0",
        #                                                                     "/yolo/segmentation_mask_image1"],
        #                        descriptor=ParameterDescriptor(
        #                                description='',
        #                                type=ParameterType.PARAMETER_STRING_ARRAY
        #                        ))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))

        self.declare_parameter(name='model_path', value="yolo11n-seg.pt",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='export_model_format', value='',
                               descriptor=ParameterDescriptor(
                                       description='Export the model to one of the supported formats '
                                                   'if the file does not exist. '
                                                   'See https://docs.ultralytics.com/modes/export/#export-formats '
                                                   'for supported formats. '
                                                   'Note that for batch inferencing, '
                                                   'the tensorrt model '
                                                   'batch={max expected stream sources, or num_cameras} '
                                                   'must be specified.',
                                       type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter('track_2d', False)
        self.declare_parameter('tracker_2d', 'bytetrack.yaml')
        self.declare_parameter('plot_tracks', True)
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter(name='show_image', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name="use_image_dimensions", value=True, descriptor=ParameterDescriptor(
                description='Whether to use the image dimensions when running inference or using a fixed square image '
                            'size for the model. Setting to True typically yields better performance.',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name="image_dimensions", value=(480, 640), descriptor=ParameterDescriptor(
                description='The image dimensions to use when running inference. '
                            'Must be set if exporting the model to another format, '
                            'e.g TensorRT .engine since that is compiled with a fixed size.',
                type=ParameterType.PARAMETER_INTEGER_ARRAY
        ))
        self.declare_parameter("resize_image", False)
        self.declare_parameter("half_precision", True)
        self.declare_parameter("conf_thresh", 0.25)
        self.declare_parameter("iou_thresh", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", ['person', 'car'])  # [] or ['person', 'car'] or [0, 2]
        self.declare_parameter('static_camera_info', True)
        self.declare_parameter('subscribe_camera_info', False)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.synchronization_interval = self.get_parameter('synchronization_interval').get_parameter_value().double_value
        self.input_image_topic_is_compressed = self.get_parameter('input_image_topic_is_compressed').get_parameter_value().bool_array_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.qos = self.get_parameter('qos').get_parameter_value().string_value
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.export_model_format = self.get_parameter('export_model_format').get_parameter_value().string_value
        self.track_2d = self.get_parameter('track_2d').get_parameter_value().bool_value
        self.tracker_2d = self.get_parameter('tracker_2d').get_parameter_value().string_value
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
        self.static_camera_info = self.get_parameter('static_camera_info').get_parameter_value().bool_value
        self.subscribe_camera_info = self.get_parameter('subscribe_camera_info').get_parameter_value().bool_value

        # Setup the device
        self.device = 'cpu'
        self.torch_device = torch.device('cpu')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.torch_device = torch.device('cuda:0')

        self.use_segmentation = "seg" in self.model_path
        self.task = "segment" if self.use_segmentation else "detect"

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
                    str(Path(self.model_path).with_suffix('.pt')),
                    # task=self.task,
            )  # can only export pytorch models
            self.model.export(
                    format=self.export_model_format, half=self.half_precision, simplify=True, nms=True,
                    # imgsz=tuple(imgsz),  # not necessary if dynamic=True
                    dynamic=True,
                    device=self.device,
                    batch=self.num_cameras
            )

            self.get_logger().info(f"Exported model to {self.export_model_format} format: {self.model_path}")
            self.model_path = str(Path(self.model_path).with_suffix('.' + self.export_model_format))

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

                export_format = Path(self.model_path).suffix.lstrip('.')
                self.model = model_class(
                        str(Path(self.model_path).with_suffix('.pt')),
                        # task=self.task,
                )  # append .pt to the model path
                self.model.export(
                        format=export_format,
                        half=self.half_precision,
                        simplify=True,
                        nms=True,
                        # imgsz=tuple(imgsz),  # not necessary if dynamic=True
                        dynamic=True,
                        device=self.device,
                        batch=self.num_cameras
                )
                self.model_path = str(Path(self.model_path).with_suffix('.' + export_format))

        # Initialize model
        self.model = model_class(
                self.model_path,
                # task=self.task,
        )

        # Filter classes
        class_names = self.model.names
        num_model_classes = len(class_names)
        class_names_inv = {v: k for k, v in class_names.items()}
        supported_class_names = set(class_names_inv.keys())
        if len(self.classes) == 0:
            self.classes = list(range(num_model_classes))
        else:
            if isinstance(self.classes, int):
                assert self.classes < num_model_classes
                self.classes = [self.classes]
            elif isinstance(self.classes, str):
                self.classes = [int(x.strip()) for x in self.classes.split(',')]  # assert all ints less than num_model_classes
                assert all(x < num_model_classes for x in self.classes)
            elif isinstance(self.classes, list):
                if isinstance(self.classes[0], str):
                    assert all(x in supported_class_names for x in self.classes)
                    self.classes = [class_names_inv[x.strip()] for x in self.classes]
            else:
                self.classes = list(self.classes)

        self.get_logger().info(f"Only detecting classes: {[class_names[class_] for class_ in self.classes]}")

        # Initialize variables
        self.bridge = CvBridge()
        self.results = None
        self.cameras = tuple(['stream_' + str(i) for i in range(self.num_cameras)])
        initial_dict = {self.cameras[idx]: None for idx in range(self.num_cameras)}
        self.camera_info = initial_dict.copy()
        self.camera_model = initial_dict.copy()
        self.images = initial_dict.copy()
        self.detection_images = initial_dict.copy()
        self.headers = initial_dict.copy()
        self.frame_ids = initial_dict.copy()
        self.msg_metadata = initial_dict.copy()
        self.image_frame_ids = initial_dict.copy()
        self.image_widths = initial_dict.copy()
        self.image_heights = initial_dict.copy()
        self.imgszs = initial_dict.copy()

        if self.plot_tracks:
            # Store the track history
            self.track_history = defaultdict(lambda: [])

        try:
            self.get_logger().info("Fusing model...")
            self.model.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fusing the model: {e}. "
                                   f"This usually occurs if not using a pytorch model (.pt), "
                                   f"e.g a TensorRT model (.engine)")

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

        # Subscribers
        self.subscriptions_ = []

        for i, camera in enumerate(self.cameras):
            self.image_message_format = "raw"
            self.image_message_type = Image
            if self.input_image_topic_is_compressed[i]:
                self.image_message_format = "compressed"
                self.image_message_type = CompressedImage

            if self.synchronization_interval > 0:
                image_subscriber = Subscriber(
                        self, self.image_message_type,
                        f"{camera}/image_raw",
                        qos_profile=qos_profile
                )
                camera_info_subscriber = Subscriber(
                        self, CameraInfo,
                        f"{camera}/camera_info",
                        qos_profile=qos_profile
                )

            else:
                # todo: setup callback groups and multithreaded executor
                image_subscriber = self.create_subscription(
                        self.image_message_type,
                        f"{camera}/image_raw",
                        lambda msg, idx=i: self.callback_common(msg, idx),
                        qos_profile=qos_profile
                )
                camera_info_subscriber = self.create_subscription(
                        CameraInfo,
                        f"{camera}/camera_info",
                        lambda msg, cam=camera: self._store_camera_info(msg, cam),
                        qos_profile=qos_profile
                )
            self.subscriptions_.append(image_subscriber)
            if self.subscribe_camera_info:
                self.subscriptions_.append(camera_info_subscriber)

        if self.synchronization_interval >= 0.0:
            if self.synchronization_interval == 0.0:
                self.ts = TimeSynchronizer(
                    self.subscriptions_, self.queue_size
                )
            elif self.synchronization_interval > 0.0:
                self.ts = ApproximateTimeSynchronizer(
                    self.subscriptions_, self.queue_size, slop=self.synchronization_interval
                )
            self.ts.registerCallback(self.image_callback_synchronized)

        else:
            # use a timer callback to synchronize images
            self.timer = self.create_timer(0.1, self.timer_callback)

        # Publishers
        self.publishers_ = []
        for i, camera in enumerate(self.cameras):
            detection_publisher = self.create_publisher(
                    Detection2DArray,
                    f"{camera}/yolo/detection/results",
                    qos_profile
            )
            self.publishers_.append(detection_publisher)

            if self.publish_debug_image:
                debug_image_publisher = self.create_publisher(
                        Image,
                        f"{camera}/yolo/detection/debug_image",
                        qos_profile
                )
                self.publishers_.append(debug_image_publisher)

                segmentation_image_publisher = self.create_publisher(
                        Image,
                        f"{camera}/yolo/detection/segmentation_image",
                        qos_profile
                )
                self.publishers_.append(segmentation_image_publisher)

                segmentation_mask_publisher = self.create_publisher(
                        Image,
                        f"{camera}/yolo/detection/segmentation_mask",
                        qos_profile
                )
                self.publishers_.append(segmentation_mask_publisher)


        self.get_logger().info(f"image_obstacle_detection_node node started on device: {self.device}")

    def _store_camera_info(self, msg, camera):
        self.camera_info[camera] = msg

    def image_callback_synchronized(self, *msg):
        for i in range(self.num_cameras):
            img_msg_idx = i if not self.subscribe_camera_info else i + 1
            self.callback_common(msg[img_msg_idx], i)

        # detect objects
        images = list(self.images.values())  # list of numpy arrays (CV images)

        # # optionally, first preprocess and convert
        # images = model.predictor.preprocess(images)  # using ultralytics preprocessor
        # images = torch.tensor(np.stack(images),
        #                       dtype=torch.float32,
        #                       device=self.torch_device).permute(0, 3, 1, 2) / 255.0  # should be the same

        self.detect_objects(images)

        for i, result in enumerate(self.results):
            detection_msg, detection_image, mask_img = self.parse_results(
                    result,
                    self.headers[self.cameras[i]])

            publisher_idx = i * 4 if self.publish_debug_image else i
            self.publishers_[publisher_idx].publish(detection_msg)
            if detection_image is not None:
                if self.publish_debug_image:
                    if self.msg_metadata[self.cameras[i]] is not None:
                        detection_image = cv2.cvtColor(
                                detection_image, self.msg_metadata[self.cameras[i]].get('inverse_conversion'))

                    detection_img_msg = self.bridge.cv2_to_imgmsg(
                            detection_image,
                            encoding=self.msg_metadata[self.cameras[i]].get('msg_fmt'))
                    detection_img_msg.header = self.headers[self.cameras[i]]
                    self.publishers_[publisher_idx + 1].publish(detection_img_msg)  # detection image

                    if mask_img is not None:
                        mask_image_msg = self.bridge.cv2_to_imgmsg(
                                mask_img,
                                encoding="mono8")
                        mask_image_msg.header = self.headers[self.cameras[i]]
                        self.publishers_[publisher_idx + 3].publish(mask_image_msg)  # segmentation mask image message

                        cv_image_inverted = cv2.cvtColor(self.images[self.cameras[i]], self.msg_metadata[self.cameras[i]].get('inverse_conversion'))
                        color_mask_img = cv2.bitwise_and(cv_image_inverted, cv_image_inverted, mask=mask_img)
                        color_mask_img_msg = self.bridge.cv2_to_imgmsg(
                                color_mask_img,
                                encoding=self.msg_metadata[self.cameras[i]].get('msg_fmt')
                        )
                        color_mask_img_msg.header = self.headers[self.cameras[i]]
                        self.publishers_[publisher_idx + 2].publish(color_mask_img_msg)


    def timer_callback(self):
        images = list(self.images.values())
        if any(img is None for img in images):
            return  # wait until every camera has delivered at least one frame

        self.detect_objects(images)

        if self.results is None:
            return

        # todo: refactor loops for speed. Use pytorch tensors where necessary to keep on device
        for i, result in enumerate(self.results):
            detection_msg, detection_image, mask_img = self.parse_results(
                    result, self.headers[self.cameras[i]])

            publisher_idx = i * 4 if self.publish_debug_image else i
            self.publishers_[publisher_idx].publish(detection_msg)

            if detection_image is not None and self.publish_debug_image:
                meta = self.msg_metadata[self.cameras[i]]
                if meta is not None:
                    detection_image = cv2.cvtColor(detection_image, meta.get('inverse_conversion'))
                detection_img_msg = self.bridge.cv2_to_imgmsg(
                        detection_image, encoding=meta.get('msg_fmt'))
                detection_img_msg.header = self.headers[self.cameras[i]]
                self.publishers_[publisher_idx + 1].publish(detection_img_msg)

                if mask_img is not None:
                    mask_image_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding="mono8")
                    mask_image_msg.header = self.headers[self.cameras[i]]
                    self.publishers_[publisher_idx + 3].publish(mask_image_msg)

                    cv_image_inv = cv2.cvtColor(self.images[self.cameras[i]], meta.get('inverse_conversion'))
                    color_mask_img = cv2.bitwise_and(cv_image_inv, cv_image_inv, mask=mask_img)
                    color_mask_img_msg = self.bridge.cv2_to_imgmsg(
                            color_mask_img, encoding=meta.get('msg_fmt'))
                    color_mask_img_msg.header = self.headers[self.cameras[i]]
                    self.publishers_[publisher_idx + 2].publish(color_mask_img_msg)


    def callback_common(self, msg, idx):
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
            if self.imgszs[self.cameras[idx]] is None:
                self.image_heights[self.cameras[idx]], self.image_widths[self.cameras[idx]] = cv_image.shape[:2]
                if self.use_image_dimensions:
                    # Check divisibility by 32 to conform to YOLOs convolution kernel size and stride length
                    new_height = self.image_heights[self.cameras[idx]] if self.image_heights[self.cameras[idx]] % 32 == 0 else ((
                                                                                                    self.image_heights[self.cameras[idx]] // 32) + 1) * 32
                    new_width = self.image_widths[self.cameras[idx]] if self.image_widths[self.cameras[idx]] % 32 == 0 else ((self.image_widths[self.cameras[idx]] // 32) + 1) * 32

                    self.imgszs[self.cameras[idx]] = (new_height, new_width)
                else:
                    self.imgszs[self.cameras[idx]] = (640, 640)

            # (optional) resize the image
            if self.resize_image and self.use_image_dimensions and (
                    (self.image_heights[self.cameras[idx]], self.image_widths[self.cameras[idx]]) != (self.imgszs[self.cameras[idx]][0], self.imgszs[self.cameras[idx]][1])):
                cv_image = cv2.resize(cv_image, (self.imgszs[self.cameras[idx]][1], self.imgszs[self.cameras[idx]][0]), interpolation=cv2.INTER_LINEAR)

            self.images[self.cameras[idx]] = cv_image
            self.headers[self.cameras[idx]] = msg.header
            self.frame_ids[self.cameras[idx]] = image_frame_id
            self.msg_metadata[self.cameras[idx]] = {
                'msg_encoding': msg_encoding,
                'msg_timestamp': msg_timestamp,  # self.get_clock().now().to_msg(),
                'msg_fmt': msg_fmt,
                'conversion': conversion,
                'inverse_conversion': inverse_conversion,
                'is_color': is_color,
                'is_depth': is_depth,
                'compressed_msg_codec': compressed_msg_codec
            }

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")
            raise e

    def detect_objects(self, image):
        try:
            imgsz = next((v for v in self.imgszs.values() if v is not None), (640, 640))
            if self.track_2d:
                # https://docs.ultralytics.com/modes/track/#why-choose-ultralytics-yolo-for-object-tracking
                self.results = self.model.track(
                        source=image,
                        conf=self.conf_thresh,
                        iou=self.iou_thresh,
                        imgsz=imgsz,
                        device=self.device,
                        half=self.half_precision,
                        classes=self.classes,
                        max_det=self.max_det,
                        retina_masks=True,
                        show=False,
                        tracker=self.tracker_2d,
                        persist=True,
                        stream=False
                )
            else:
                # https://docs.ultralytics.com/modes/predict/#inference-arguments
                self.results = self.model.predict(
                        source=image,
                        conf=self.conf_thresh,
                        iou=self.iou_thresh,
                        imgsz=imgsz,
                        device=self.device,
                        half=self.half_precision,
                        classes=self.classes,
                        max_det=self.max_det,
                        retina_masks=True,
                        show=False,
                        stream=False
                )

                # # or control each step (predict/track does all three steps)
                # im = model.predictor.preprocess(source)[0]
                # preds = model.predictor.model(im)  # inference only
                # results = model.predictor.postprocess(preds)  # post-processing
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            self.results = None

    def parse_results(self, results, header):
        if results is not None:
            detections_msg, mask_img, detection_image = self.create_detections_array(results, header)

            return detections_msg, detection_image, mask_img

    def create_detections_array(self, result, header):
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = header.stamp  # self.get_clock().now().to_msg()
        detections_msg.header.frame_id = header.frame_id

        mask_img = None

        detection_image = result.plot()
        if self.show_image:
            # Visualize the results on the frame
            cv2.imshow("image", detection_image)
            cv2.waitKey(1)
        bounding_box = result.boxes.cpu()  # Boxes object for bounding box outputs. n x 4
        classes = result.boxes.cls.cpu()  # n,
        confidence_score = result.boxes.conf.cpu()  # n,
        masks = result.masks
        keypoints = result.keypoints
        obb = result.obb
        probs = result.probs

        if bounding_box.shape[0] < 1:
            return detections_msg, mask_img, detection_image

        track_ids = None
        if self.track_2d:
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
                cv2.polylines(detection_image, [points],
                              isClosed=False, color=(230, 230, 230), thickness=5)

            # pack 2D detection results
            detection_2d = pack_2d_detection(
                bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)), conf,
                    id=track_ids[i] if track_ids is not None else -1)
            detections_msg.detections.append(detection_2d)

        return detections_msg, mask_img, detection_image

def main(args=None):
    rclpy.init(args=args)
    node = MultiStreamDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()