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

from ultralytics.engine.results import Boxes
from ultralytics.trackers.basetrack import BaseTrack
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_requirements, check_yaml


class TrackingNode(Node):
    def __init__(self):
        super(TrackingNode, self).__init__('tracking_node_2d')
        self.declare_parameter('tracker_2d', 'bytetrack.yaml')
        self.declare_parameter('fps', 30)
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('synchronization_interval', 0.1)
        self.declare_parameter('use_gpu', True)
        self.declare_parameter(name='show_image', value=False, descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_BOOL))
        self.declare_parameter(name='qos', value="SENSOR_DATA", descriptor=ParameterDescriptor(
                description='',
                type=ParameterType.PARAMETER_STRING))

        # Get parameters
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.tracker_2d = self.get_parameter('tracker_2d').value
        self.fps = self.get_parameter('fps').value
        self.queue_size = self.get_parameter('queue_size').value
        self.synchronization_interval = self.get_parameter('synchronization_interval').value
        self.use_gpu = self.get_parameter('use_gpu').value
        self.show_image = self.get_parameter('show_image').value
        self.qos = self.get_parameter('qos').value

        # Create the tracker
        self.tracker = self.create_tracker(self.tracker_2d, fps=self.fps, use_gpu=self.use_gpu)

        # Initialize variables
        self.bridge = CvBridge()

        # Setup QoS
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
        self.image_sub = Subscriber(self, Image, "image_raw", qos_profile=qos_profile)
        self.detections_sub = Subscriber(self, Detection2DArray, "detections_2d", qos_profile=qos_profile)

        self.ts = ApproximateTimeSynchronizer((self.image_sub, self.detections_sub), self.queue_size, slop=self.synchronization_interval)
        self.ts.registerCallback(self.detection_callback)

        # Initialize publishers
        self.tracked_detections_pub = self.create_publisher(Detection2DArray, "tracked_detections_2d", qos_profile=qos_profile)


        self.get_logger().info("Tracking node started")


    def detection_callback(self, img_msg, detections_msg):
        tracked_detections_msg = Detection2DArray()
        tracked_detections_msg.header = img_msg.header

        try:
            cv_image = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"Error converting image to cv2: {e}")
            return

        # parse detections
        detection_list = []
        for detection in detections_msg.detections:
            # convert YOLO class ID from string to int

            detection_list.append(
                    [
                        detection.bbox.center.position.x - detection.bbox.size_x / 2,
                        detection.bbox.center.position.y - detection.bbox.size_y / 2,
                        detection.bbox.center.position.x + detection.bbox.size_x / 2,
                        detection.bbox.center.position.y + detection.bbox.size_y / 2,
                        detection.results[0].hypothesis.score,
                        int(detection.id),  # detection.results[0].hypothesis.class_id,
                    ]
            )

        # tracking
        if len(detection_list) > 0:

            det = Boxes(np.array(detection_list), (img_msg.height, img_msg.width))
            tracks = self.tracker.update(det, cv_image)

            if len(tracks) > 0:

                for t in tracks:

                    tracked_box = Boxes(t[:-1], (img_msg.height, img_msg.width))

                    # IoU match: find the input detection whose box best overlaps this tracked box.
                    # t[-1] is tracker-internal class index, NOT a detection list index.
                    tracked_xyxy = tracked_box.xyxy[0].tolist()
                    best_iou, best_idx = 0.0, 0
                    for j, det_msg in enumerate(detections_msg.detections):
                        cx, cy = det_msg.bbox.center.position.x, det_msg.bbox.center.position.y
                        hw, hh = det_msg.bbox.size_x / 2, det_msg.bbox.size_y / 2
                        det_xyxy = [cx - hw, cy - hh, cx + hw, cy + hh]
                        xi1 = max(tracked_xyxy[0], det_xyxy[0])
                        yi1 = max(tracked_xyxy[1], det_xyxy[1])
                        xi2 = min(tracked_xyxy[2], det_xyxy[2])
                        yi2 = min(tracked_xyxy[3], det_xyxy[3])
                        inter = max(0.0, xi2 - xi1) * max(0.0, yi2 - yi1)
                        area_t = (tracked_xyxy[2] - tracked_xyxy[0]) * (tracked_xyxy[3] - tracked_xyxy[1])
                        area_d = det_msg.bbox.size_x * det_msg.bbox.size_y
                        union = area_t + area_d - inter
                        iou = inter / union if union > 0 else 0.0
                        if iou > best_iou:
                            best_iou, best_idx = iou, j
                    tracked_detection = detections_msg.detections[best_idx]

                    # get boxes values
                    box = tracked_box.xywh[0]
                    tracked_detection.bbox.center.position.x = float(box[0])
                    tracked_detection.bbox.center.position.y = float(box[1])
                    tracked_detection.bbox.size_x = float(box[2])
                    tracked_detection.bbox.size_y = float(box[3])

                    # get track id
                    track_id = ""
                    if tracked_box.is_track:
                        track_id = str(int(tracked_box.id))
                    tracked_detection.results[0].hypothesis.class_id = track_id  # replace class_id with track_id
                    tracked_detection.id = track_id

                    # append msg
                    tracked_detections_msg.detections.append(tracked_detection)

        # publish detections
        self.tracked_detections_pub.publish(tracked_detections_msg)

    def create_tracker(self, tracker_yaml: str, fps: int = 30, use_gpu: bool = True) -> BaseTrack:
        TRACKER_MAP = {"bytetrack": BYTETracker, "botsort": BOTSORT}
        check_requirements("lap")  # for linear_assignment

        tracker = check_yaml(tracker_yaml)
        cfg = IterableSimpleNamespace(**yaml_load(tracker))

        assert cfg.tracker_type in [
            "bytetrack",
            "botsort",
        ], f"Only 'bytetrack' and 'botsort' are supported for now, but got '{cfg.tracker_type}'"
        if use_gpu and torch.cuda.is_available():
            cfg.device = "0"
        else:
            cfg.device = "cpu"
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=fps)
        return tracker

def main(args=None):
    rclpy.init(args=args)
    node = TrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down node...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()