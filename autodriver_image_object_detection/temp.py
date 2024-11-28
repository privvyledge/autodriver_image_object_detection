#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import Detection3DArray, Detection3D, BoundingBox3D
from visualization_msgs.msg import MarkerArray, Marker
from cv_bridge import CvBridge
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import message_filters
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
import tf2_ros
from tf2_geometry_msgs import do_transform_pose
import sensor_msgs.point_cloud2 as pc2
from sklearn.cluster import DBSCAN


class YOLOv8TrackingNode(Node):
    def __init__(self):
        super().__init__('yolov8_3d_tracking_node')

        # Initialize parameters
        self.declare_parameter('model_path', 'yolov8x-seg.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('use_depth', True)  # True for depth image, False for pointcloud
        self.declare_parameter('camera_frame', 'camera_color_optical_frame')

        # Load parameters
        model_path = self.get_parameter('model_path').value
        self.conf_threshold = self.get_parameter('confidence_threshold').value
        self.use_depth = self.get_parameter('use_depth').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # Initialize YOLO model
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Initialize TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize subscribers
        if self.use_depth:
            self.rgb_sub = message_filters.Subscriber(
                    self, Image, '/camera/color/image_raw')
            self.depth_sub = message_filters.Subscriber(
                    self, Image, '/camera/aligned_depth_to_color/image_raw')
            self.ts = message_filters.ApproximateTimeSynchronizer(
                    [self.rgb_sub, self.depth_sub], 10, 0.1)
            self.ts.registerCallback(self.image_callback)
        else:
            self.rgb_sub = message_filters.Subscriber(
                    self, Image, '/camera/color/image_raw')
            self.cloud_sub = message_filters.Subscriber(
                    self, PointCloud2, '/camera/depth/color/points')
            self.ts = message_filters.ApproximateTimeSynchronizer(
                    [self.rgb_sub, self.cloud_sub], 10, 0.1)
            self.ts.registerCallback(self.pointcloud_callback)

        # Initialize publishers
        self.detections_pub = self.create_publisher(
                Detection3DArray, '/object_detections', 10)
        self.markers_pub = self.create_publisher(
                MarkerArray, '/detection_markers', 10)

        self.get_logger().info('YOLOv8 3D tracking node initialized')

    def image_callback(self, rgb_msg, depth_msg):
        """Process RGB and depth image pairs"""
        try:
            # Convert ROS messages to OpenCV images
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, "32FC1")

            # Run YOLOv8 inference
            results = self.model.track(rgb_image, conf=self.conf_threshold, persist=True)[0]

            # Process detections
            detections_3d = Detection3DArray()
            detections_3d.header = rgb_msg.header
            markers = MarkerArray()

            if results.boxes is not None:
                for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                    # Get 2D bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = box.cls[0].cpu().numpy()
                    track_id = box.id[0].cpu().numpy() if box.id is not None else -1

                    # Get mask
                    if mask is not None:
                        mask = mask.cpu().numpy()

                        # Get 3D points from depth image for masked region
                        masked_depth = cv2.bitwise_and(depth_image, depth_image, mask=mask.astype(np.uint8))
                        points_3d = self.depth_to_3d(masked_depth)

                        if len(points_3d) > 0:
                            # Calculate 3D bounding box
                            bbox3d = self.calculate_3d_bbox(points_3d)

                            # Create Detection3D message
                            detection = self.create_detection_msg(
                                    bbox3d, confidence, class_id, track_id)
                            detections_3d.detections.append(detection)

                            # Create visualization marker
                            marker = self.create_marker(bbox3d, i, track_id)
                            markers.markers.append(marker)

            # Publish results
            self.detections_pub.publish(detections_3d)
            self.markers_pub.publish(markers)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def pointcloud_callback(self, rgb_msg, cloud_msg):
        """Process RGB and pointcloud pairs"""
        try:
            # Convert ROS message to OpenCV image
            rgb_image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")

            # Run YOLOv8 inference
            results = self.model.track(rgb_image, conf=self.conf_threshold, persist=True)[0]

            # Convert pointcloud to numpy array
            points = np.array(list(pc2.read_points(cloud_msg,
                                                   field_names=("x", "y", "z"),
                                                   skip_nans=True)))

            # Process detections
            detections_3d = Detection3DArray()
            detections_3d.header = rgb_msg.header
            markers = MarkerArray()

            if results.boxes is not None:
                for i, (box, mask) in enumerate(zip(results.boxes, results.masks)):
                    if mask is not None:
                        # Project mask to pointcloud
                        mask = mask.cpu().numpy()
                        masked_points = self.mask_pointcloud(points, mask, cloud_msg)

                        if len(masked_points) > 0:
                            # Cluster points
                            clusters = self.cluster_points(masked_points)

                            for cluster in clusters:
                                # Calculate 3D bounding box
                                bbox3d = self.calculate_3d_bbox(cluster)

                                # Create Detection3D message
                                confidence = box.conf[0].cpu().numpy()
                                class_id = box.cls[0].cpu().numpy()
                                track_id = box.id[0].cpu().numpy() if box.id is not None else -1

                                detection = self.create_detection_msg(
                                        bbox3d, confidence, class_id, track_id)
                                detections_3d.detections.append(detection)

                                # Create visualization marker
                                marker = self.create_marker(bbox3d, i, track_id)
                                markers.markers.append(marker)

            # Publish results
            self.detections_pub.publish(detections_3d)
            self.markers_pub.publish(markers)

        except Exception as e:
            self.get_logger().error(f'Error processing pointcloud: {str(e)}')

    def depth_to_3d(self, depth_image):
        """Convert depth image to 3D points"""
        # Camera intrinsic parameters (should be loaded from camera_info)
        fx = 525.0  # focal length x
        fy = 525.0  # focal length y
        cx = 319.5  # optical center x
        cy = 239.5  # optical center y

        # Get indices of valid depth pixels
        rows, cols = np.where(depth_image > 0)

        if len(rows) == 0:
            return np.array([])

        # Convert depth to 3D points
        z = depth_image[rows, cols]
        x = (cols - cx) * z / fx
        y = (rows - cy) * z / fy

        return np.column_stack((x, y, z))

    def mask_pointcloud(self, points, mask, cloud_msg):
        """Apply 2D mask to 3D pointcloud"""
        # Project 3D points to 2D image plane
        # This requires camera intrinsics and extrinsics
        # Simplified version - assumes points are already in camera frame
        fx = 525.0
        fy = 525.0
        cx = 319.5
        cy = 239.5

        # Project points to image
        x_2d = (points[:, 0] * fx / points[:, 2]) + cx
        y_2d = (points[:, 1] * fy / points[:, 2]) + cy

        # Find points that project into the mask
        valid_points = (x_2d >= 0) & (x_2d < mask.shape[1]) & \
                       (y_2d >= 0) & (y_2d < mask.shape[0])

        x_2d = x_2d[valid_points].astype(int)
        y_2d = y_2d[valid_points].astype(int)

        # Get points that fall within the mask
        masked_points = points[valid_points][mask[y_2d, x_2d] > 0]

        return masked_points

    def cluster_points(self, points, eps=0.05, min_samples=10):
        """Cluster 3D points using DBSCAN"""
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
        clusters = []

        for label in set(clustering.labels_):
            if label != -1:  # Ignore noise points
                cluster_points = points[clustering.labels_ == label]
                clusters.append(cluster_points)

        return clusters

    def calculate_3d_bbox(self, points):
        """Calculate 3D bounding box from points"""
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        center = (min_coords + max_coords) / 2
        dimensions = max_coords - min_coords

        return {
            'center': center,
            'dimensions': dimensions,
            'orientation': np.array([0, 0, 0, 1])  # Default quaternion
        }

    def create_detection_msg(self, bbox3d, confidence, class_id, track_id):
        """Create Detection3D message"""
        detection = Detection3D()

        # Set bounding box
        detection.bbox.center.position = Point(
                x=float(bbox3d['center'][0]),
                y=float(bbox3d['center'][1]),
                z=float(bbox3d['center'][2])
        )

        detection.bbox.center.orientation = Quaternion(
                x=float(bbox3d['orientation'][0]),
                y=float(bbox3d['orientation'][1]),
                z=float(bbox3d['orientation'][2]),
                w=float(bbox3d['orientation'][3])
        )

        detection.bbox.size = Vector3(
                x=float(bbox3d['dimensions'][0]),
                y=float(bbox3d['dimensions'][1]),
                z=float(bbox3d['dimensions'][2])
        )

        # Set metadata
        detection.id = str(track_id)
        detection.score = float(confidence)
        detection.tracking_id = str(track_id)

        return detection

    def create_marker(self, bbox3d, marker_id, track_id):
        """Create visualization marker"""
        marker = Marker()
        marker.header.frame_id = self.camera_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "detection"
        marker.id = marker_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD

        # Set pose
        marker.pose.position = Point(
                x=float(bbox3d['center'][0]),
                y=float(bbox3d['center'][1]),
                z=float(bbox3d['center'][2])
        )
        marker.pose.orientation = Quaternion(
                x=float(bbox3d['orientation'][0]),
                y=float(bbox3d['orientation'][1]),
                z=float(bbox3d['orientation'][2]),
                w=float(bbox3d['orientation'][3])
        )

        # Set scale
        marker.scale = Vector3(
                x=float(bbox3d['dimensions'][0]),
                y=float(bbox3d['dimensions'][1]),
                z=float(bbox3d['dimensions'][2])
        )

        # Set color (using track_id to generate unique colors)
        color_hash = hash(str(track_id))
        marker.color.r = float((color_hash & 0xFF0000) >> 16) / 255.0
        marker.color.g = float((color_hash & 0x00FF00) >> 8) / 255.0
        marker.color.b = float(color_hash & 0x0000FF) / 255.0
        marker.color.a = 0.5

        return marker


def main(args=None):
    rclpy.init(args=args)
    node = YOLOv8TrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()