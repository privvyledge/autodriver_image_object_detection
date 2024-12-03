"""
ROS2 node for obstacle detection using Euclidean Clustering on Point Clouds.
Publishes clustered pointcloud, visualization markers, and object array.

Usage:
    sudo apt-get install ros-${ROS_DISTRO}-derived-object-msgs ros-${ROS_DISTRO}-vision-msgs
    ros2 run pointcloud_obstacle_detection euclidean_clustering_node

1. Subscribe to image, pointcloud, depth using message filters
2. Detect obstacles using YOLO [done]
3. Publish detections to vision_msgs/Detection2DArray and Image [done]
4. Implement projection to 3D [done]

Todo:
    * Implement non-tracking (i.e predict) [done]
    * Setup segmentation (https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Masks | )
        * show segmentation mask image [done]
        * publish segmentation mask image [done]
        * resize the mask to the original shape (or depth image shape) [done: no need]
    * Setup 3D [depth] [done]
    * transform the depth masks/bboxes to the base_link frame [done]
    * Setup 3D [pointcloud]
    * transform the pointcloud masks/bboxes to the base_link frame [done]
    * add pre transformed points to the pointcloud to avoid multiple transformations and see if it improves performance [done: it does]
    * filter out objects/clusters with min_height above a certain threshold
    * publish as a derived_object
    * rename rgb to visual
    * add check for .engine model with try-catch and then convert to tensorrt
    * switch to engine models as the default
    * publish detection pointcloud for debugging
    * plot tracks over time
    * Setup OBB (https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.OBB)
"""

import time
import struct
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
from ultralytics import YOLO

try:
    import open3d as o3d
    import open3d.core as o3c
except ImportError:
    pass


class ImageObstacleDetectionNode(Node):
    def __init__(self):
        super(ImageObstacleDetectionNode, self).__init__("image_obstacle_detection_node")

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
        self.declare_parameter(name='depth_image_topic', value="/camera/camera/aligned_depth_to_color/image_raw",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='depth_camera_info_topic',
                               value="/camera/camera/aligned_depth_to_color/camera_info",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='pointcloud_topic',
                               value="/camera/camera/depth/color/points",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='model_path', value="yolov8m-seg.pt",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('track_2d', True)
        self.declare_parameter('track_3d', True)
        self.declare_parameter('tracker_2d', 'bytetrack.yaml')
        self.declare_parameter('queue_size', 1)
        self.declare_parameter('synchronization_interval', 0.1)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter(name='show_image', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter("use_image_dimensions", True)
        self.declare_parameter("resize_image", False)
        self.declare_parameter("half_precision", True)
        self.declare_parameter("conf_thresh", 0.25)
        self.declare_parameter("iou_thresh", 0.5)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", list(range(80)))
        self.declare_parameter("project_to_3d", True)  # todo: remove this flag and just use depth or pointcloud
        self.declare_parameter("use_depth", True)  # can run both at the same time at the cost of speed. Depth is significantly faster for now (about 2.5 times)
        self.declare_parameter("use_pointcloud", False)  # can run both at the same time at the cost of speed. Depth is significantly faster for now (about 2.5 times)
        self.declare_parameter("output_frame", "base_link")  # e.g base_link.
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter("transform_timeout", 0.1)
        self.declare_parameter('static_camera_info', True)
        self.declare_parameter('depth_scale', 1000.0)
        self.declare_parameter('depth_max', 4.0)
        self.declare_parameter(name='normalize_depth', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='normalized_max', value=255, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_DOUBLE))

        self.declare_parameter('crop_to_roi', True)
        self.declare_parameter('roi_min', [-6.0, -6.0, 0.0])
        self.declare_parameter('roi_max', [6.0, 6.0, 2.0])
        self.declare_parameter('voxel_size', 0.05)  # 0.01, 0.05
        self.declare_parameter('remove_statistical_outliers', False)
        self.declare_parameter('estimate_normals', False)
        self.declare_parameter('remove_ground', False)
        self.declare_parameter("cluster_tolerance", 0.2)  # meters
        self.declare_parameter("min_cluster_size", 100)
        self.declare_parameter("bounding_box_type", "AABB")  # AABB or OBB

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
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
        self.track_2d = self.get_parameter('track_2d').value
        self.track_3d = self.get_parameter('track_3d').value
        self.tracker_2d = self.get_parameter('tracker_2d').value
        self.queue_size = self.get_parameter('queue_size').value
        self.synchronization_interval = self.get_parameter('synchronization_interval').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.show_image = self.get_parameter('show_image').get_parameter_value().bool_value
        self.use_image_dimensions = self.get_parameter("use_image_dimensions").get_parameter_value().bool_value
        self.resize_image = self.get_parameter("resize_image").get_parameter_value().bool_value
        self.half_precision = self.get_parameter("half_precision").get_parameter_value().bool_value
        self.conf_thresh = self.get_parameter("conf_thresh").get_parameter_value().double_value
        self.iou_thresh = self.get_parameter("iou_thresh").get_parameter_value().double_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.classes = (
            self.get_parameter("classes").get_parameter_value().integer_array_value
        )
        self.project_to_3d = self.get_parameter("project_to_3d").get_parameter_value().bool_value
        self.use_depth = self.get_parameter("use_depth").get_parameter_value().bool_value
        self.use_pointcloud = self.get_parameter("use_pointcloud").get_parameter_value().bool_value
        self.output_frame = self.get_parameter("output_frame").value
        self.static_camera_to_robot_tf = self.get_parameter(
            "static_camera_to_robot_tf").get_parameter_value().bool_value
        self.transform_timeout = self.get_parameter("transform_timeout").get_parameter_value().double_value
        self.static_camera_info = self.get_parameter("static_camera_info").get_parameter_value().bool_value
        self.depth_scale = self.get_parameter("depth_scale").get_parameter_value().double_value
        self.depth_max = self.get_parameter("depth_max").get_parameter_value().double_value
        self.normalize_depth = self.get_parameter("normalize_depth").get_parameter_value().bool_value
        self.normalized_max = self.get_parameter("normalized_max").get_parameter_value().double_value
        self.crop_to_roi = self.get_parameter('crop_to_roi').value
        self.roi_min = self.get_parameter('roi_min').value
        self.roi_max = self.get_parameter('roi_max').value
        self.voxel_size = self.get_parameter('voxel_size').value
        self.remove_statistical_outliers = self.get_parameter('remove_statistical_outliers').value
        self.estimate_normals = self.get_parameter('estimate_normals').value
        self.remove_ground = self.get_parameter('remove_ground').value
        self.cluster_tolerance = self.get_parameter("cluster_tolerance").get_parameter_value().double_value
        self.min_cluster_size = self.get_parameter("min_cluster_size").get_parameter_value().integer_value
        self.bounding_box_type = self.get_parameter("bounding_box_type").value

        # Setup the device
        self.device = 'cpu'
        self.torch_device = torch.device('cpu')
        self.o3d_device = o3d.core.Device('CPU:0')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.torch_device = torch.device('cuda:0')
                if o3d.core.cuda.is_available():
                    self.o3d_device = o3d.core.Device('CUDA:0')

        # Initialize variables
        self.image_frame_id = None
        self.image_width = None
        self.image_height = None
        self.imgsz = None
        self.bridge = CvBridge()
        self.model = YOLO(self.model_path)
        self.use_segmentation = self.model_path.endswith("-seg.pt")
        self.results = None
        self.detection_image = None
        self.cameras = ['rgb', 'depth'] if (self.use_depth and self.project_to_3d) else ['rgb']
        self.camera_infos = {'rgb': None, 'depth': None}
        self.camera_models = {'rgb': PinholeCameraModel(),
                              'depth': PinholeCameraModel()}
        self.o3d_camera_intrinsics = {'rgb': None,
                                      'depth': None} if self.use_depth else {'rgb': None}
        self.o3d_camera_models = {'rgb': o3d.camera.PinholeCameraIntrinsic(), 'depth': o3d.camera.PinholeCameraIntrinsic()}
        self.images = {'rgb': None, 'depth': None, 'rgbd': None, 'pointcloud': None}  # todo: rename to frame
        self.frame_ids = {'rgb': None, 'depth': None, 'rgbd': None, 'pointcloud': None}
        self.headers = {'rgb': None, 'depth': None, 'rgbd': None, 'pointcloud': None}
        self.msg_metadata = {'rgb': None, 'depth': None, 'rgbd': None, 'pointcloud': None}
        if self.project_to_3d and self.use_pointcloud:
            self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)
        self.previous_time = time.time()
        self.previous_callback_time = None
        self.last_timestamp = None
        self.camera_to_robot_tf = None
        self.camera_to_robot_tf_o3d = None
        self.previous_rgbd_image = None
        self.previous_pointcloud = None

        try:
            self.get_logger().info("Fusing model...")
            self.model.fuse()
        except TypeError as e:
            self.get_logger().warn(f"Error while fuse: {e}. "
                                   f"This usually occurs if not using a pytorch model (.pt), "
                                   f"e.g a TensorRT model (.engine)")

        # Initialize TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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

        self.ts = ApproximateTimeSynchronizer(self.subscriptions_, self.queue_size, slop=0.1)
        self.ts.registerCallback(self.detection_callback)

        # Publishers
        # self.cluster_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)
        self.detection_results_pub = self.create_publisher(Detection2DArray, self.detection_results_topic,
                                                           self.queue_size)
        if self.project_to_3d:
            if self.use_depth:
                self.detection3d_depth_results_pub = self.create_publisher(Detection3DArray,
                                                                           "/yolo/detection3d_depth_results",
                                                                           self.queue_size)
                self.marker_depth_pub = self.create_publisher(MarkerArray,
                                                              '/detection3d_depth_markers',
                                                              self.queue_size)
            if self.use_pointcloud:
                self.detection3d_pointcloud_results_pub = self.create_publisher(Detection3DArray,
                                                                           "/yolo/detection3d_pointcloud_results",
                                                                           self.queue_size)
                self.marker_pointcloud_pub = self.create_publisher(MarkerArray,
                                                              '/detection3d_pointcloud_markers',
                                                              self.queue_size)
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

        # # Timers
        # self.timer = self.create_timer(0.1, self.timer_callback)
        # self.timer_count = 0

        self.get_logger().info(f"image_obstacle_detection_node node started on device: {self.device}")

    def detection_callback(self, *msg):
        """
        Todo: If the input to YOLO is not a Pytorch Tensor, the array is converted to RGB using numpy.
        Therefore, manually transfer the image to a torch tensor and perform operations (e.g color conversion and resizing) using torch before passing to YOLO to reduce CPU utilization. """
        # Save/update camera infos
        for camera in self.cameras:
            # initialize the camera infos and models
            if self.camera_infos[camera] is None:
                self.camera_infos[camera] = msg[1] if camera == 'rgb' else msg[3]
                self.camera_models[camera].fromCameraInfo(self.camera_infos[camera])
                self.o3d_camera_intrinsics[camera] = self.convert_to_open3d_tensor(self.camera_models[camera].K)

            # update the camera infos and models if not static
            if not self.static_camera_info:
                self.camera_infos[camera] = msg[1] if camera == 'rgb' else msg[3]
                self.camera_models[camera].fromCameraInfo(self.camera_infos[camera])
                self.o3d_camera_intrinsics[camera] = self.convert_to_open3d_tensor(self.camera_models[camera].K)

        try:
            msg_timestamp = None
            msg_fmt = "bgr8"
            conversion = None
            inverse_conversion = None
            is_color = True
            is_depth = False
            for camera in self.cameras:
                img_msg = msg[0] if camera == 'rgb' else msg[2]
                cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt, conversion, inverse_conversion, is_color, is_depth = self.parse_image_message(img_msg)
                self.frame_ids[camera] = image_frame_id
                self.headers[camera] = img_msg.header
                self.msg_metadata[camera] = {'msg_encoding': msg_encoding, 'msg_timestamp': msg_timestamp,
                                             'msg_fmt': msg_fmt, 'conversion': conversion,
                                             'inverse_conversion': inverse_conversion,
                                             'is_color': is_color, 'is_depth': is_depth}

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
                            self.imgsz = (640, 640)

                    # (optional) resize the image
                    if self.resize_image and self.use_image_dimensions and (
                            (self.image_height, self.image_width) != (self.imgsz[0], self.imgsz[1])):
                        cv_image = cv2.resize(cv_image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)

                # update the image dictionary
                self.images[camera] = cv_image

            # detect/track objects in the visual image
            self.detect_objects(self.images['rgb'])

            if self.project_to_3d and self.use_pointcloud:
                # Clear the pointcloud
                self.o3d_pointcloud.clear()
                # unpack pointcloud message
                ros_cloud = msg[-1]
                pc_frame_id, pc_msg_timestamp, pc_fields = self.unpack_pointcloud_message(ros_cloud)

                if pc_frame_id is not None:
                    self.frame_ids['pointcloud'] = pc_frame_id
                    self.headers['pointcloud'] = ros_cloud.header
                    self.msg_metadata['pointcloud'] = {'msg_timestamp': pc_msg_timestamp, 'field_names': pc_fields}
                    self.images['pointcloud'] = self.o3d_pointcloud.point.positions.cpu().numpy()

            # try:
            #    self.results = next(self.results)
            # except TypeError:
            #    pass
            detection_msg, detection_image, mask_img = self.parse_results(self.results, self.headers['rgb'])

            # publish the detection array results
            self.detection_results_pub.publish(detection_msg)

            # convert OpenCV image back to the input msg_fmt
            if detection_image is not None:
                if self.msg_metadata['rgb'].get('conversion', None) is not None:
                    detection_image = cv2.cvtColor(detection_image, self.msg_metadata['rgb'].get('inverse_conversion'))

                if self.image_message_format in ("compressed", "packet"):
                    detection_image_msg = self.bridge.compressed_imgmsg_to_cv2(
                            detection_image,
                            desired_encoding=self.msg_metadata['rgb'].get('msg_fmt'))
                else:
                    detection_image_msg = self.bridge.cv2_to_imgmsg(
                            detection_image,
                            encoding=self.msg_metadata['rgb'].get('msg_fmt'))

                detection_image_msg.header.frame_id = self.frame_ids['rgb']
                detection_image_msg.header.stamp = self.headers['rgb'].stamp

                if self.publish_debug_image:
                    if self.detection_image_topic:
                        self.detection_image_pub.publish(detection_image_msg)
                    if self.segmentation_mask_image_topic and (mask_img is not None):
                        if self.image_message_format in ("compressed", "packet"):
                            mask_image_msg = self.bridge.compressed_imgmsg_to_cv2(
                                    mask_img,
                                    desired_encoding="mono8")
                        else:
                            mask_image_msg = self.bridge.cv2_to_imgmsg(
                                    mask_img,
                                    encoding="mono8")

                        mask_image_msg.header.frame_id = self.frame_ids['rgb']
                        mask_image_msg.header.stamp = self.headers['rgb'].stamp
                        self.segmentation_mask_image_pub.publish(mask_image_msg)

                    if self.segmentation_image_topic and (mask_img is not None):
                        # color_mask_img = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
                        color_mask_img = cv2.bitwise_and(self.images['rgb'], self.images['rgb'], mask=mask_img)
                        if self.image_message_format in ("compressed", "packet"):
                            color_mask_image_msg = self.bridge.compressed_imgmsg_to_cv2(
                                    color_mask_img,
                                    desired_encoding=self.msg_metadata['rgb'].get('msg_fmt'))
                        else:
                            color_mask_image_msg = self.bridge.cv2_to_imgmsg(
                                    color_mask_img,
                                    encoding=self.msg_metadata['rgb'].get('msg_fmt'))

                        color_mask_image_msg.header.frame_id = self.frame_ids['rgb']
                        color_mask_image_msg.header.stamp = self.headers['rgb'].stamp
                        self.segmentation_image_pub.publish(color_mask_image_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def parse_image_message(self, msg):
        image_frame_id = msg.header.frame_id
        msg_timestamp = msg.header.stamp
        msg_encoding = msg.encoding
        msg_fmt = "bgr8"
        conversion = None
        inverse_conversion = None
        is_color = True
        is_depth = False

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
                is_color, is_depth)

    def unpack_pointcloud_message(self, ros_cloud):
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

    def preprocess_pointcloud(self, frame_id, timestamp):
        # transform to robot frame
        self.get_camera_to_robot_tf(frame_id, timestamp)

        if self.camera_to_robot_tf_o3d is not None:
            # copy the original positions before transforming for later use without inversion.
            # Leads to significant speedup over multiple transformations.
            self.o3d_pointcloud.point.positions_inv = self.o3d_pointcloud.point.positions.clone()
            self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf_o3d)
            frame_id = self.output_frame

        ## Remove duplicate points
        #start_time = time.time()
        #mask = self.o3d_pointcloud.remove_duplicated_points()
        #self.o3d_pointcloud = self.o3d_pointcloud.select_by_mask(mask)
        #self.processing_times['remove_duplicate_points'] = time.time() - start_time

        ## Remove NaN points
        #start_time = time.time()
        #self.o3d_pointcloud = self.o3d_pointcloud.remove_non_finite_points(remove_nan=True, remove_infinite=True)
        #self.processing_times['remove_nan_points'] = time.time() - start_time

        # ROI cropping
        if self.crop_to_roi:
            # points_o3d = points_o3d.crop(self.roi_min, self.roi_max)
            mask = (
                    (self.o3d_pointcloud.point.positions[:, 0] >= self.roi_min[0]) &
                    (self.o3d_pointcloud.point.positions[:, 0] <= self.roi_max[0]) &
                    (self.o3d_pointcloud.point.positions[:, 1] >= self.roi_min[1]) &
                    (self.o3d_pointcloud.point.positions[:, 1] <= self.roi_max[1]) &
                    (self.o3d_pointcloud.point.positions[:, 2] >= self.roi_min[2]) &
                    (self.o3d_pointcloud.point.positions[:, 2] <= self.roi_max[2])
            )
            self.o3d_pointcloud = self.o3d_pointcloud.select_by_mask(mask)

        # Voxel downsampling
        if self.voxel_size > 0.0:
            self.o3d_pointcloud = self.o3d_pointcloud.voxel_down_sample(self.voxel_size)

        if self.remove_statistical_outliers:
            self.o3d_pointcloud, _ = self.o3d_pointcloud.remove_statistical_outliers(nb_neighbors=20, std_ratio=2.0)

        if self.estimate_normals:
            self.o3d_pointcloud.estimate_normals(
                    radius=0.1,  # Use a radius of 10 cm for local geometry
                    max_nn=30  # Use up to 30 nearest neighbors
            )

        # Ground segmentation.
        if self.remove_ground:
            plane_model, inliers = self.o3d_pointcloud.segment_plane(
                    distance_threshold=0.2,
                    ransac_n=5,
                    num_iterations=100
            )
            # ground_cloud = self.o3d_pointcloud.select_by_index(inliers)  # ground
            self.o3d_pointcloud = self.o3d_pointcloud.select_by_index(inliers, invert=True)  #

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
                        # classes=self.classes,
                        # max_det=self.max_det,
                        retina_masks=True,
                        show=False,
                        tracker=self.tracker_2d,
                        persist=True,
                        stream=True
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
                        # classes=self.classes,
                        # max_det=self.max_det,
                        retina_masks=True,
                        show=False,
                        stream=True
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
        detections_msg.header.stamp = self.msg_metadata['rgb'].get('msg_timestamp')  # self.get_clock().now().to_msg()
        detections_msg.header.frame_id = self.frame_ids['rgb']

        mask_img = None

        if self.project_to_3d:
            # get transform
            if self.use_depth:
                depth_timestamp = self.msg_metadata['depth'].get(
                    'msg_timestamp') if self.output_frame else self.msg_metadata['rgb'].get(
                    'msg_timestamp')  # self.get_clock().now().to_msg()

                if depth_timestamp is None:
                    depth_timestamp = self.get_clock().now().to_msg()

                self.get_camera_to_robot_tf(self.frame_ids['depth'], depth_timestamp)
                detection3d_depth_array = Detection3DArray()
                detection3d_depth_array.header.frame_id = self.output_frame if self.output_frame else self.frame_ids['depth']
                detection3d_depth_array.header.stamp = depth_timestamp

                marker_depth_array = MarkerArray()

            # Preprocess pointcloud: if input is a pointcloud, transform the pointcloud here
            if self.use_pointcloud:
                pointcloud_timestamp = self.msg_metadata['pointcloud'].get(
                    'msg_timestamp') if self.output_frame else self.msg_metadata['rgb'].get(
                    'msg_timestamp')  # self.get_clock().now().to_msg()

                if pointcloud_timestamp is None:
                    pointcloud_timestamp = self.get_clock().now().to_msg()

                self.preprocess_pointcloud(
                    self.frame_ids['pointcloud'], pointcloud_timestamp)
                detection3d_pointcloud_array = Detection3DArray()
                detection3d_pointcloud_array.header.frame_id = self.output_frame if self.output_frame else self.frame_ids[
                    'pointcloud']
                detection3d_pointcloud_array.header.stamp = pointcloud_timestamp

                marker_pointcloud_array = MarkerArray()


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

            if self.track_2d:
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()  # todo: add to message

            if hasattr(result, "masks") and masks is not None:
                # mask_data = masks.data.cpu()  # masks drawn on the image [0-1] float. n x image_height x image_width
                # mask_orig_shape = masks.orig_shape  # tuple [image_width, image_height]
                # masks_shape = masks.shape
                # masks_xy = masks.xy  # list of size n, each item (ndarray) of size [m, 2], where m is the number of pixels per object mask
                mask_img = (torch.sum(masks.data, dim=0).cpu().numpy() * 255).astype(np.uint8)
                if self.show_image:
                    cv2.imshow("masked_image", mask_img)
                    cv2.waitKey(1)

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
                mask_xy = mask.xy[0]
                track_id = box.id.int().cpu().item() if box.id is not None else -1

                # pack 2D detection results
                detection_2d = self.pack_2d_detection(
                    bbox[0], bbox[1], bbox[2], bbox[3], result.names.get(int(cls)), conf)
                detections_msg.detections.append(detection_2d)

                if self.project_to_3d:
                    # could run depth and pointcloud processing in different threads
                    if self.use_depth and (self.images['depth'] is not None):
                        x, y, z, size_x, size_y, size_z, quat, points_3d = self.project_to_3d_with_depth(
                            mask, bbox, self.images['depth'])

                        if x is None:
                            continue

                        # transform the boxes to the robot frame
                        frame_id = self.frame_ids["depth"]
                        if self.output_frame:
                            frame_id = self.output_frame
                            if self.camera_to_robot_tf is not None:
                                x, y, z, size_x, size_y, size_z, points_3d = self.transform_bbox_3d(
                                    x, y, z, size_x, size_y, size_z, points_3d)

                        # Append 3D detection (x, y, z, size_x, size_y, size_z, confidence, class_id)
                        detection3d_depth_array.detections.append(
                            self.create_3d_detection(x, y, z, size_x, size_y, size_z, conf, result.names.get(int(cls))))
                        marker_depth_array.markers.append(
                            self.create_marker(
                                i, x, y, z, size_x, size_y, size_z, frame_id,
                                depth_timestamp, conf, result.names.get(int(cls)), track_id=None))

                    if self.use_pointcloud and (self.images['pointcloud'] is not None):
                        x, y, z, size_x, size_y, size_z, quat, points_3d = self.project_to_3d_with_pointcloud(
                            mask, bbox, self.images['pointcloud'])

                        if x is None:
                            continue

                        # no need to transform the boxes to the robot frame since the pointcloud is transformed
                        frame_id = self.frame_ids["pointcloud"] if not self.output_frame else self.output_frame

                        # Append 3D detection (x, y, z, size_x, size_y, size_z, confidence, class_id)
                        detection3d_pointcloud_array.detections.append(
                            self.create_3d_detection(
                                x, y, z, size_x, size_y, size_z, conf, result.names.get(int(cls)), quat)
                        )
                        marker_pointcloud_array.markers.append(
                            self.create_marker(
                                i, x, y, z, size_x, size_y, size_z, frame_id,
                                pointcloud_timestamp, conf, result.names.get(int(cls)), track_id=None, quat=quat))

            if self.project_to_3d:
                if self.use_depth:
                    self.detection3d_depth_results_pub.publish(detection3d_depth_array)
                    self.marker_depth_pub.publish(marker_depth_array)

                if self.use_pointcloud:
                    self.detection3d_pointcloud_results_pub.publish(detection3d_pointcloud_array)
                    self.marker_pointcloud_pub.publish(marker_pointcloud_array)
        return detections_msg, mask_img

    def pack_2d_detection(self, x, y, size_x, size_y, class_id, conf):
        detection = Detection2D()
        detection.bbox.center.position.x = float(x)
        detection.bbox.center.position.y = float(y)
        detection.bbox.size_x = float(size_x)
        detection.bbox.size_y = float(size_y)
        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_id
        hypothesis.hypothesis.score = float(conf)
        detection.results.append(hypothesis)
        return detection

    def project_to_3d_with_depth(self, mask, xywh, depth_image):
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
        bbox_center_x, bbox_center_y = map(int, xywh[:2])
        bbox_size_x, bbox_size_y = map(int, xywh[2:])

        mask_data = mask.data.cpu().numpy().astype(np.uint8)[0, :, :] * 255
        mask_xy = mask.xy[0]

        # Step 2: Get the ROI of the mask (or bbox) in the depth image
        if mask is not None:
            # crop depth image by mask
            # mask_array = np.array(
            #     [[int(ele[0]), int(ele[1])] for ele in mask_xy]
            # )
            # mask_ = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            # cv2.fillPoly(mask_, [np.array(mask_array, dtype=np.int32)], 255)
            # roi = cv2.bitwise_and(depth_image, depth_image, mask=mask_)  # same as below

            roi = cv2.bitwise_and(depth_image,
                                   depth_image,
                                   mask=mask_data)  # same as above

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

        # filter out z-values above the depth max
        rows, cols = np.where(roi > 0)
        depths = roi[rows, cols]

        # find the z coordinate on the 3D BB
        if mask is not None:
            # Step 4: filter out invalid depth pixels, i.e roi[roi > 0]
            roi = roi[roi > 0]

            # Step 5: Find the z coordinate (distance) of the mask, np.median(roi) or bounding_box_center/depth_scale
            bb_center_z_coord = np.median(roi)
            ''' Method 2: z1 = np.median(valid_depths)  # Approximate depth. np.median(mask_depth_values)'''

        else:
            bb_center_z_coord = (
                    depth_image[bbox_center_y][bbox_center_x] / self.depth_scale
            )

        # Step 6: (optional) crop values outside of depth_max
        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= self.depth_max
        if not np.any(mask_z):
            return None, None, None, None, None, None, None, None

        roi = roi[mask_z]
        z_min, z_max = np.min(roi), np.max(roi)
        z = (z_max + z_min) / 2

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

    def project_to_3d_with_pointcloud(self, mask, xywh, pointcloud=None):
        if self.o3d_pointcloud.is_empty() and pointcloud is None:
            # Ensure the point cloud is in memory
            self.get_logger().info(f"Pointcloud is empty")
            return None, None, None, None, None, None, None, None

        bbox_center_x, bbox_center_y = map(int, xywh[:2])
        bbox_size_x, bbox_size_y = map(int, xywh[2:])
        u, v = bbox_center_x, bbox_center_y  # xywh[0:2] or np.mean(mask_indices, axis=0)
        cx, cy = self.camera_models["rgb"].cx(), self.camera_models["rgb"].cy()
        fx, fy = self.camera_models["rgb"].fx(), self.camera_models['rgb'].fy()

        mask_data = mask.data[0, :, :]
        mask_data = (o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(mask_data)).to(o3c.Dtype.Int64) * 255)
        mask_xy = mask.xy[0]

        points = self.o3d_pointcloud.point.positions

        # project back to the camera frame if the points were projected to the robots frame to project to the image
        if self.output_frame and self.camera_to_robot_tf_o3d is not None:
            # self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf_o3d.inv())
            points = self.o3d_pointcloud.point.positions_inv

        # Step 1: Project 3D points to 2D image plane
        x_2d = (points[:, 0] * fx / points[:, 2]) + cx
        y_2d = (points[:, 1] * fy / points[:, 2]) + cy

        # Step 2: project the mask to the pointcloud
        valid_points_mask = (x_2d >= 0) & (x_2d < mask_data.shape[1]) & \
                       (y_2d >= 0) & (y_2d < mask_data.shape[0])  # Find points that project into the mask image

        # Filter valid projected points
        # valid_projected_points = self.o3d_pointcloud.point.positions[valid_points_mask]
        valid_projected_points = self.o3d_pointcloud.select_by_mask(valid_points_mask)

        if valid_projected_points.is_empty():
            return None, None, None, None, None, None, None, None

        valid_x_2d = x_2d[valid_points_mask].to(o3c.Dtype.Int64)
        valid_y_2d = y_2d[valid_points_mask].to(o3c.Dtype.Int64)

        # Get mask values for these points.
        mask_values = mask_data[valid_y_2d, valid_x_2d]

        # Select points that fall within the mask
        # masked_points = valid_projected_points[mask_values > 0]
        masked_points = valid_projected_points.select_by_mask(mask_values > 0)

        # Step 3: cluster points and get bounding boxes. todo: move this outside of the loop to perform once
        clusters, bboxes = self.cluster_points(masked_points)
        # only select the cluster with the largest label
        # get the center (if not using the bbox)
        # centroid = np.mean(clustered_points, axis=0)
        if bboxes:
            for bbox in bboxes:
                if self.bounding_box_type.lower() == "obb":
                    center = bbox.center.cpu().numpy().tolist()
                    extent = bbox.extent.cpu().numpy().tolist()
                    # Convert rotation matrix to quaternion
                    R = bbox.rotation.cpu().numpy()
                    quat = quaternion_from_matrix(np.vstack((np.hstack((R, [[0], [0], [0]])), [0, 0, 0, 1])))
                    quat = Quaternion(x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

                else:
                    center = bbox.get_center().cpu().numpy().tolist()
                    extent = bbox.get_extent().cpu().numpy().tolist()
                    quat = None
                x, y, z = center
                size_x, size_y, size_z = extent
            return x, y, z, size_x, size_y, size_z, quat, clusters
        return None, None, None, None, None, None, None, None

    def cluster_points(self, o3d_pcd):
        """
        Perform DBSCAN clustering and return clusters with their bounding boxes.
        Return the list of clusters.
        Todo: remove clusters with more points than self.max_cluster_size
        """
        # self.get_logger().info("Clustering point cloud with DBSCAN...")

        # Ensure the point cloud is in memory
        if o3d_pcd.is_empty():
            self.get_logger().info(f"Pointcloud is empty")
            return [], []

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

        if len(unique_labels) == 0:
            self.get_logger().info(f"unique labels: {unique_labels}, len: {len(unique_labels)}")
            return [], []  # Return empty lists if no valid clusters

        # unique_labels = unique_labels[unique_labels >= 0]

        max_label = labels.max().item()
        self.get_logger().info(f"DBSCAN found {max_label + 1} clusters")

        clusters = []
        bboxes = []

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

            # # Get cluster height. todo: use mask
            # points = cluster_pcd.point.positions.cpu().numpy()
            # min_z = np.min(points[:, 2])
            # max_z = np.max(points[:, 2])
            # height = max_z - min_z
            #
            # # Filter clusters by height
            # if height < self.cluster_min_height or height > self.cluster_max_height:
            #     continue

            clusters.append(cluster_pcd)

            if self.bounding_box_type.lower() == "aabb":
                bounding_box = cluster_pcd.get_axis_aligned_bounding_box()
            elif self.bounding_box_type.lower() == "obb":
                bounding_box = cluster_pcd.get_oriented_bounding_box()
            else:
                raise ValueError(f"Unknown bounding box type: {self.bounding_box_type}")

            bboxes.append(bounding_box)

        return clusters, bboxes

    def create_3d_detection(self, x, y, z, size_x, size_y, size_z, confidence, class_id, quat=None):
        det_msg = Detection3D()
        det_msg.bbox.center.position.x = float(x)
        det_msg.bbox.center.position.y = float(y)
        det_msg.bbox.center.position.z = float(z)
        if quat is not None:
            det_msg.bbox.center.orientation = quat
        det_msg.bbox.size.x = float(size_x)  # Width
        det_msg.bbox.size.y = float(size_y)  # Height
        det_msg.bbox.size.z = float(size_z) # Depth approximation

        hypothesis = ObjectHypothesisWithPose()
        hypothesis.hypothesis.class_id = class_id
        hypothesis.hypothesis.score = float(confidence)
        det_msg.results.append(hypothesis)
        return det_msg

    def create_marker(self, marker_id, x, y, z, size_x, size_y, size_z,
                      frame_id, timestamp=None, confidence=None, class_id=None, track_id=None, quat=None):
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
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.5

        marker.lifetime = rclpy.duration.Duration(seconds=0.1).to_msg()
        return marker

    def get_camera_to_robot_tf(self, source_frame_id, timestamp=None):
        if self.camera_to_robot_tf is not None and self.static_camera_to_robot_tf:
            return

        if timestamp is None:
            timestamp = rclpy.time.Time()
        if self.output_frame:
            # Try to get the transform from camera to robot
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.output_frame,
                    source_frame_id,
                    timestamp,  # this could also be the depth msg timestamp
                    rclpy.duration.Duration(seconds=self.transform_timeout)
                )
            except tf2_ros.LookupException as e:
                self.get_logger().error(f"TF Lookup Error: {str(e)}")
                return
            except tf2_ros.ConnectivityException as e:
                self.get_logger().error(f"TF Connectivity Error: {str(e)}")
                return
            except tf2_ros.ExtrapolationException as e:
                self.get_logger().error(f"TF Extrapolation Error: {str(e)}")
                return

            # Convert the TF transform to a 4x4 transformation matrix
            self.camera_to_robot_tf = self.transform_to_matrix(transform)
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

    def transform_bbox_3d(self, x, y, z, size_x, size_y, size_z, points_3d):
        # transform the pose
        object_pose_camera_frame = np.array([x, y, z, 1])
        object_pose_robot_frame = np.dot(self.camera_to_robot_tf, object_pose_camera_frame)
        x_robot, y_robot, z_robot = object_pose_robot_frame[:3]

        # transform the size
        object_size_camera_frame = np.array([size_x, size_y, size_z, 1])
        object_size_robot_frame = np.dot(self.camera_to_robot_tf, object_size_camera_frame)
        size_x_robot, size_y_robot, size_z_robot = object_pose_robot_frame[:3]

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
            input_array = o3c.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(input_array))

        return input_array

    def normalize_depth_image(self, depth_image, max_val=255, dtype=None):
        """

        :param depth_image:
        :param max_val: Either 1 or 255
        :param dtype: Either cv2.CV_32F or cv2.CV_8UC1
        :return:
        """
        if self.use_gpu or (self.cpu_backend == 'torch'):
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

    def destroy_node(self):
        cv2.destroyAllWindows()
        del self.model
        if self.project_to_3d and self.use_pointcloud:
            del self.o3d_pointcloud
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()


def main(args=None):
    rclpy.init(args=args)
    node = ImageObstacleDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
