"""
Single stream detector for object detection with optional tracking.
Usage:
    sudo apt-get install ros-${ROS_DISTRO}-vision-msgs
    ros2 run autodriver_image_object_detection single_stream_detector
"""

import time
import uuid
import struct
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

def pack_derived_object_msg(x, y, size_x, size_y, class_id, conf, id=None):
    """Convert a nav2_dynamic_msgs/Obstacle into a derived_object_msgs/Object."""
    obj = Object()
    # Convert first 4 bytes of the obstacle's UUID into a uint32 id.
    if id in (None, -1):
        id = 10000000
    obj.id = id

    # Set detection level. Here we assume that an obstacle from tracking
    # is equivalent to a TRACKED object.
    obj.detection_level = Object.OBJECT_TRACKED

    # Set pose. Use the obstacle's position and set a default orientation.
    obj.pose.position = Point(x=float(x), y=float(y), z=0.0)
    obj.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    # Set twist using the obstacle's velocity; angular part is set to zero.
    obj.twist.linear = Vector3(x=0.0, y=0.0, z=0.0)
    obj.twist.angular = Vector3(x=0.0, y=0.0, z=0.0)

    # Set acceleration to zero (no info available).
    obj.accel.linear = Vector3(x=0.0, y=0.0, z=0.0)
    obj.accel.angular = Vector3(x=0.0, y=0.0, z=0.0)

    # Leave polygon empty.
    # Convert obstacle size into a SolidPrimitive shape (assuming a box).
    sp = SolidPrimitive()
    sp.type = SolidPrimitive.BOX
    sp.dimensions = [float(size_x), float(size_y), 0.0]
    obj.shape = sp

    # Set classification fields to defaults.
    obj.classification = {
        'car': Object.CLASSIFICATION_CAR,
        'truck': Object.CLASSIFICATION_TRUCK,
        'bus': Object.CLASSIFICATION_OTHER_VEHICLE,
        'person': Object.CLASSIFICATION_PEDESTRIAN,
        'motorcycle': Object.CLASSIFICATION_MOTORCYCLE,
        'bicycle': Object.CLASSIFICATION_BIKE,
        'train': Object.CLASSIFICATION_OTHER_VEHICLE,
        'airplane': Object.CLASSIFICATION_UNKNOWN_BIG,
        'boat': Object.CLASSIFICATION_UNKNOWN_MEDIUM,
    }.get(class_id, Object.CLASSIFICATION_UNKNOWN)  # Object.CLASSIFICATION_UNKNOWN_SMALL

    # Mark the object as classified if the detection score is high.
    obj.object_classified = bool(conf > 0.25)  # same as conf_threshold
    
    # Convert the obstacle score (0-1) to a certainty value (0-255)
    obj.classification_certainty = int(conf * 255)
    obj.classification_age = 0

    return obj


class SingleStreamDetector(Node):
    def __init__(self):
        super(SingleStreamDetector, self).__init__("single_stream_detector")

        # Declare parameters
        self.declare_parameter(name='input_image_topic', value="/camera/camera/color/image_raw",
                               descriptor=ParameterDescriptor(
                                       description='The input image topic. '
                                                   'Works with all image types: RGB(A), BGR(A), mono8, mono16.',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='input_camera_info_topic', value="/camera/camera/color/camera_info",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='input_image_topic_is_compressed', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='detection_results_topic', value="/yolo/detection_results",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('publish_debug_image', True)
        self.declare_parameter(name='detection_image_topic', value="/yolo/detection_image",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='segmentation_image_topic', value="/yolo/segmentation_image",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='segmentation_mask_image_topic', value="/yolo/segmentation_mask_image",
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
        self.declare_parameter("iou_thresh", 0.7)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", ['person', 'car'])  # [] or ['person', 'car'] or [0, 2]
        self.declare_parameter('static_camera_info', True)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.input_image_topic = self.get_parameter('input_image_topic').get_parameter_value().string_value
        self.input_camera_info_topic = self.get_parameter('input_camera_info_topic').get_parameter_value().string_value
        self.input_image_topic_is_compressed = self.get_parameter('input_image_topic_is_compressed').get_parameter_value().bool_value
        self.detection_results_topic = self.get_parameter('detection_results_topic').get_parameter_value().string_value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.detection_image_topic = self.get_parameter('detection_image_topic').get_parameter_value().string_value
        self.segmentation_image_topic = self.get_parameter('segmentation_image_topic').get_parameter_value().string_value
        self.segmentation_mask_image_topic = self.get_parameter('segmentation_mask_image_topic').get_parameter_value().string_value
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

        # Setup the device
        self.device = 'cpu'
        self.torch_device = torch.device('cpu')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.torch_device = torch.device('cuda:0')

        # Initialize variables
        self.image_frame_id = None
        self.image_width = None
        self.image_height = None
        self.imgsz = None
        self.bridge = CvBridge()

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
            self.model = model_class(self.model_path.split('.')[0] + '.pt')  # can only export pytorch models
            self.model.export(
                    format=self.export_model_format, half=self.half_precision, simplify=True, nms=True,
                    # imgsz=tuple(imgsz),  # not necessary if dynamic=True
                    dynamic=True,
                    device=self.device
            )

            self.get_logger().info(f"Exported model to {self.export_model_format} format: {self.model_path}")
            self.model_path = self.model_path.split('.')[:-1] + '.' + self.export_model_format

        # if model_path ends with .engine or .onnx, try loading the file and export if FileNotFoundError
        if self.model_path.split('.')[-1] in ['engine', 'onnx']:
            try:
                self.model = model_class(self.model_path)
            except FileNotFoundError:
                self.get_logger().info(f"Model not found: {self.model_path}. "
                                       f"Trying to export to {self.model_path.split('.')[-1]}.")

                self.model = model_class(self.model_path.split('.')[0] + '.pt')  # append .pt to the model path
                self.model.export(
                        format=self.model_path.split('.')[-1],
                        half=self.half_precision,
                        simplify=True,
                        nms=True,
                        # imgsz=tuple(imgsz),  # not necessary if dynamic=True
                        dynamic=True,
                        device=self.device
                )
                self.model_path = self.model_path.split('.')[0] + '.' + self.model_path.split('.')[-1]

        # Initialize model
        self.model = model_class(self.model_path)

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
                    self.classes = [class_names_inv[x.strip()] for x in self.classes]  # todo: assert that all are in class_names
            else:
                self.classes = list(self.classes)

        self.get_logger().info(f"Only detecting classes: {[class_names[class_] for class_ in self.classes]}")

        self.use_segmentation = self.model_path.endswith("-seg.pt")
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
                qos_profile)

        self.object_array_pub = self.create_publisher(
                ObjectArray,
                'yolo/objects',
                qos_profile
        )

        try:
            self.obstacle_detection_pub = self.create_publisher(ObstacleArray, 'yolo/obstacles', qos_profile)
        except NameError:
            pass

        if self.publish_debug_image:
            self.detection_image_pub = self.create_publisher(
                    self.image_message_type,
                    self.detection_image_topic,
                    qos_profile)
            self.segmentation_image_pub = self.create_publisher(
                    self.image_message_type,
                    self.segmentation_image_topic,
                    qos_profile)
            self.segmentation_mask_image_pub = self.create_publisher(
                    self.image_message_type,
                    self.segmentation_mask_image_topic,
                    qos_profile)

        self.get_logger().info(
            (
                f"single_stream_detector started. "
                f"Publishing on {self.detection_results_topic}. "
                f"Subscribing to {self.input_image_topic}."
            )
        )

    def image_callback(self, msg):
        try:
            msg_timestamp = None
            msg_fmt = "bgr8"
            conversion = None
            inverse_conversion = None
            is_color = True
            is_depth = False
            cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt, conversion, inverse_conversion, is_color, is_depth, compressed_msg_codec = self.parse_image_message(
                msg)

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
                    self.imgsz = (640, 640)

            # (optional) resize the image
            if self.resize_image and self.use_image_dimensions and (
                    (self.image_height, self.image_width) != (self.imgsz[0], self.imgsz[1])):
                cv_image = cv2.resize(cv_image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)

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
                        # color_mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
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
                                    encoding=msg_fmt)

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

    def parse_image_message(self, msg):
        image_frame_id = msg.header.frame_id
        msg_timestamp = msg.header.stamp
        msg_fmt = "bgr8"
        compressed_msg_codec = None
        conversion = None
        inverse_conversion = None
        is_color = True
        is_depth = False
        if self.image_message_format == "raw":
            msg_encoding = msg.encoding

        elif self.image_message_format == 'compressed':
            # format: rgb8; jpeg compressed bgr8
            msg_info = msg.format
            msg_encoding_split = msg_info.split(';')
            uncompressed_msg_fmt = msg_encoding_split[0]
            compressed_img_info = msg_encoding_split[1].split()
            compressed_msg_codec = compressed_img_info[0]
            msg_encoding = compressed_img_info[-1]

        # set the desired output encoding
        # (http://wiki.ros.org/cv_bridge/Tutorials/UsingCvBridgeToConvertBetweenROSImagesAndOpenCVImages#cv_bridge.2FTutorials.2FUsingCvBridgeCppDiamondback.Converting_ROS_image_messages_to_OpenCV_images)
        if (msg_encoding.find("mono8") != -1) or (msg_encoding.find("8UC1") != -1):
            msg_fmt = "mono8"  # "8UC1"
            is_color = False
            conversion = cv2.COLOR_GRAY2BGR
            inverse_conversion = cv2.COLOR_BGR2GRAY
        elif msg_encoding.find("bgra") != -1:
            msg_fmt = "bgra8"  # "8UC4"
            conversion = cv2.COLOR_BGRA2BGR
            inverse_conversion = cv2.COLOR_BGR2BGRA
        elif msg_encoding.find("rgba") != -1:
            msg_fmt = "rgba8"  # "8UC4"
            conversion = cv2.COLOR_RGBA2BGR
            inverse_conversion = cv2.COLOR_BGR2RGBA
        elif msg_encoding.find("bgr8") != -1:
            msg_fmt = "bgr8"  # or 8UC3
            # conversion = cv2.COLOR_BGR2BGR
            # inverse_conversion = cv2.COLOR_BGR2BGR
        elif (msg_encoding.find("rgb8") != -1):
            msg_fmt = "rgb8"  # or 8UC3
            conversion = cv2.COLOR_RGB2BGR
            inverse_conversion = cv2.COLOR_BGR2RGB
        elif msg_encoding.find("16UC1") != -1:
            msg_fmt = "16UC1"  # "16UC1", mono16
            is_color = False
            is_depth = True
            # raise NotImplementedError("Depth images are not supported for YOLO detection")
            #conversion = cv2.COLOR_GRAY2BGR
            #inverse_conversion = cv2.COLOR_BGR2GRAY
        else:
            self.get_logger().error("Unsupported encoding:", msg_encoding)
            self.exit(1)

        # convert ROS2 image message to OpenCV
        if self.image_message_format in ("compressed", "packet"):
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, msg_fmt)
        else:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg_fmt)

        if conversion is not None:
            cv_image = cv2.cvtColor(cv_image, conversion)
        return (cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt, conversion, inverse_conversion,
                is_color, is_depth, compressed_msg_codec)

    def detect_objects(self, image):
        try:
            if self.track_2d:
                # https://docs.ultralytics.com/modes/track/#why-choose-ultralytics-yolo-for-object-tracking
                self.results = self.model.track(
                        source=image,
                        conf=self.conf_thresh,
                        iou=self.iou_thresh,
                        imgsz=self.imgsz,
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
                        imgsz=self.imgsz,
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
            detections_msg, mask_img = self.create_detections_array(results, header)

            return detections_msg, self.detection_image, mask_img

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
            self.detection_image = result.plot()
            if self.show_image:
                # Visualize the results on the frame
                cv2.imshow("image", self.detection_image)
                cv2.waitKey(1)
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
                        bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)), conf, id=track_ids[i] if track_ids is not None else -1)
                    obstacle_msg.obstacles.append(obstacle_2d)


            self.object_array_pub.publish(objects_msg)

            if obstacle_msg is not None:
                self.obstacle_detection_pub.publish(obstacle_msg)

            return detections_msg, mask_img


def main(args=None):
    rclpy.init(args=args)
    node = SingleStreamDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()