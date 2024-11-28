"""
ROS2 node for obstacle detection using Euclidean Clustering on Point Clouds.
Publishes clustered pointcloud, visualization markers, and object array.

Usage:
    sudo apt-get install ros-${ROS_DISTRO}-derived-object-msgs ros-${ROS_DISTRO}-vision-msgs
    ros2 run pointcloud_obstacle_detection euclidean_clustering_node

1. Subscribe to image, pointcloud, depth using message filters
2. Detect obstacles using YOLO [done]
3. Publish detections to vision_msgs/Detection2DArray and Image [done]
4. Implement projection to 3D

* Implement non-tracking (i.e predict) [done]
* Setup 3D [depth]
* Setup 3D [pointcloud]
* Setup segmentation (https://docs.ultralytics.com/reference/engine/results/#ultralytics.engine.results.Masks | )
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
from message_filters import Subscriber, ApproximateTimeSynchronizer
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


class ImageObstacleDetectionNode(Node):
    def __init__(self):
        super(ImageObstacleDetectionNode, self).__init__("image_obstacle_detection_node")

        # Declare parameters
        self.declare_parameter(name='input_image_topic', value="/camera/camera/color/image_raw",
                               descriptor=ParameterDescriptor(
                                       description='The input image topic. '
                                                   'Works with all image types: RGB(A), BGR(A), mono8, mono16.',
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
        self.declare_parameter(name='segmentation_image_topic', value="/yolo/segmentation_mask",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))
        self.declare_parameter(name='model_path', value="yolov8n-seg.pt",
                               descriptor=ParameterDescriptor(
                                       description='',
                                       type=ParameterType.PARAMETER_STRING))
        self.declare_parameter('track_2d', True)
        self.declare_parameter('track_3d', True)
        self.declare_parameter('tracker_2d', 'bytetrack.yaml')
        self.declare_parameter('queue_size', 1)
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
        self.declare_parameter("project_to_3d", True)
        self.declare_parameter("use_depth", True)  # True for depth image, False for pointcloud
        self.declare_parameter("output_frame", "base_link")
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter("transform_timeout", 0.1)

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').get_parameter_value().bool_value
        self.input_image_topic = self.get_parameter('input_image_topic').value
        self.input_image_topic_is_compressed = self.get_parameter('input_image_topic_is_compressed').value
        self.detection_results_topic = self.get_parameter('detection_results_topic').value
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value
        self.detection_image_topic = self.get_parameter('detection_image_topic').value
        self.segmentation_image_topic = self.get_parameter('segmentation_image_topic').value
        self.qos = self.get_parameter('qos').value
        self.model_path = self.get_parameter('model_path').value
        self.track_2d = self.get_parameter('track_2d').value
        self.track_3d = self.get_parameter('track_3d').value
        self.tracker_2d = self.get_parameter('tracker_2d').value
        self.queue_size = self.get_parameter('queue_size').value
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
        self.output_frame = self.get_parameter("output_frame").value
        self.static_camera_to_robot_tf = self.get_parameter("static_camera_to_robot_tf").get_parameter_value().bool_value
        self.transform_timeout = self.get_parameter("transform_timeout").get_parameter_value().double_value

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
        self.cameras = ['rgb', 'depth']
        self.camera_infos = {'rgb': None, 'depth': None}
        self.camera_models = {'rgb': PinholeCameraModel(), 'depth': PinholeCameraModel()}
        self.o3d_camera_intrinsics = {'rgb': None,
                                      'depth': None}
        # self.o3d_camera_models = {'rgb': o3d.camera.PinholeCameraIntrinsic(), 'depth': o3d.camera.PinholeCameraIntrinsic()}
        self.images = {'rgb': None, 'depth': None, 'rgbd': None}
        self.depth_frame = ""
        self.previous_time = time.time()
        self.previous_callback_time = None
        self.last_timestamp = None
        self.camera_to_robot_tf = None
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

        self.message_format = "raw"
        self.message_type = Image
        if self.input_image_topic_is_compressed or "compressed" in self.input_image_topic:
            self.message_format = "compressed"
            self.message_type = CompressedImage

        # Subscribers
        self.image_sub = self.create_subscription(self.message_type, self.input_image_topic,
                                                  self.image_callback, qos_profile=qos_profile)


        # Publishers
        # self.cluster_pub = self.create_publisher(PointCloud2, self.output_topic, self.queue_size)
        self.detection_results_pub = self.create_publisher(Detection2DArray, self.detection_results_topic,
                                                           self.queue_size)
        if self.publish_debug_image and self.detection_image_topic:
            self.detection_image_pub = self.create_publisher(self.message_type,
                                                             self.detection_image_topic, self.queue_size)

        # # Timers
        # self.timer = self.create_timer(0.1, self.timer_callback)
        # self.timer_count = 0

        self.get_logger().info(f"image_obstacle_detection_node node started on device: {self.device}")

    def image_callback(self, msg):
        """
        Todo: If the input to YOLO is not a Pytorch Tensor, the array is converted to RGB using numpy.
        Therefore, manually transfer the image to a torch tensor and perform operations (e.g color conversion and resizing) using torch before passing to YOLO to reduce CPU utilization. """
        try:
            if self.image_frame_id is None:
                self.image_frame_id = msg.header.frame_id
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
                msg_fmt = "mono16"  # "16UC1"
                is_color = False
                is_depth = True
                # raise NotImplementedError("Depth images are not supported for YOLO detection")
                conversion = cv2.COLOR_GRAY2BGR
                inverse_conversion = cv2.COLOR_BGR2GRAY
            else:
                self.get_logger().error("Unsupported encoding:", msg_encoding)
                self.exit(1)

            # convert ROS2 image message to OpenCV
            if self.message_format in ("compressed", "packet"):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, msg_fmt)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding=msg_fmt)

            if conversion is not None:
                cv_image = cv2.cvtColor(cv_image, conversion)

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
            if self.resize_image and self.use_image_dimensions and ((height, width) != (new_height, new_width)):
                cv_image = cv2.resize(image, (self.imgsz[1], self.imgsz[0]), interpolation=cv2.INTER_LINEAR)

            # detect/track objects
            self.detect_objects(cv_image)

            detection_msg, detection_image = self.parse_results(self.results)

            # convert OpenCV image back to the input msg_fmt
            if conversion is not None:
                detection_image = cv2.cvtColor(detection_image, inverse_conversion)

            self.detection_results_pub.publish(detection_msg)

            if self.message_format in ("compressed", "packet"):
                detection_image_msg = self.bridge.compressed_imgmsg_to_cv2(detection_image, desired_encoding=msg_fmt)
            else:
                detection_image_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding=msg_fmt)

            detection_image_msg.header.frame_id = self.image_frame_id
            detection_image_msg.header.stamp = msg.header.stamp
            self.detection_image_pub.publish(detection_image_msg)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

    def detect_objects(self, image):
        try:
            if self.track_2d:
                # https://docs.ultralytics.com/modes/track/#why-choose-ultralytics-yolo-for-object-tracking
                self.results = self.model.track(
                        source=image,
                        conf=self.conf_thresh,
                        iou=self.iou_thresh,
                        imgsz=self.imgsz,
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
                        half=self.half_precision,
                        # classes=self.classes,
                        # max_det=self.max_det,
                        retina_masks=True,
                        show=False,
                        stream=True
                )
        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')
            self.results = None


    def parse_results(self, results):
        if results is not None:
            detections_msg = self.create_detections_array(results)

            return detections_msg, self.detection_image

    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = self.get_clock().now().to_msg()
        detections_msg.header.frame_id = self.image_frame_id

        masks_msg = []

        for result in results:
            self.detection_image = result.plot()
            if self.show_image:
                # Visualize the results on the frame
                cv2.imshow("image", self.detection_image)
                cv2.waitKey(1)
            bounding_box = result.boxes.xywh.cpu()  # Boxes object for bounding box outputs
            classes = result.boxes.cls.cpu()
            confidence_score = result.boxes.conf.cpu()
            masks = result.masks
            keypoints = result.keypoints
            obb = result.obb
            probs = result.probs

            if self.track_2d:
                track_ids = result.boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()

            if masks is not None:
                masks = masks.cpu().numpy()  # Masks object for segmentation masks outputs

                if hasattr(result, "masks") and masks is not None:
                    for mask_tensor in masks:
                        mask_numpy = (
                                np.squeeze(mask_tensor).astype(
                                        np.uint8
                                )
                                * 255
                        )
                        mask_image_msg = self.bridge.cv2_to_imgmsg(
                                mask_numpy, encoding="mono8"
                        )
                        masks_msg.append(mask_image_msg)

            if keypoints is not None:
                keypoints = keypoints.cpu()  # Keypoints object for pose outputs

            if obb is not None:
                obb = obb.cpu()  # Oriented boxes object for OBB outputs

            if probs is not None:
                probs = probs.cpu()  # Probs object for classification outputs

            for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
                detection = Detection2D()
                detection.bbox.center.position.x = float(bbox[0])
                detection.bbox.center.position.y = float(bbox[1])
                detection.bbox.size_x = float(bbox[2])
                detection.bbox.size_y = float(bbox[3])
                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = result.names.get(int(cls))
                hypothesis.hypothesis.score = float(conf)
                detection.results.append(hypothesis)
                detections_msg.detections.append(detection)
        return detections_msg, masks_msg

    def parse_masks(self):
        pass

    def destroy_node(self):
        cv2.destroyAllWindows()
        del self.model
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
