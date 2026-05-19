"""Depth fusion node: lift Detection2DArray to 3D using depth images or LiDAR pointclouds."""
import numpy as np
import torch
import torch.utils.dlpack

try:
    from tf_transformations import quaternion_matrix, quaternion_from_matrix
    _TF_TRANSFORMS = True
except ImportError:
    _TF_TRANSFORMS = False

try:
    import open3d as o3d
    import open3d.core as o3c
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

import rclpy
import rclpy.duration
import rclpy.time
from rclpy.executors import ExternalShutdownException
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from cv_bridge import CvBridge
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from vision_msgs.msg import Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Quaternion
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_ros import TransformListener, Buffer

from autodriver_image_object_detection.base_detector import BaseDetector
from autodriver_image_object_detection.utils.pointcloud_utils import (
    unpack_pointcloud_message, create_marker,
)


def _quaternion_to_matrix(q):
    """Fallback: [x, y, z, w] → 4×4 rotation matrix (tf_transformations not installed)."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),   2*(x*z + y*w), 0.0],
        [  2*(x*y + z*w), 1 - 2*(x*x + z*z),   2*(y*z - x*w), 0.0],
        [  2*(x*z - y*w),   2*(y*z + x*w), 1 - 2*(x*x + y*y), 0.0],
        [            0.0,             0.0,             0.0, 1.0],
    ], dtype=np.float64)


class DepthFusionNode(BaseDetector):
    """Lift Detection2DArray to 3D using depth images and/or LiDAR pointclouds.

    Subscribes to Detection2DArray from any 2D detector plus depth/pointcloud sensor
    topics, performs 3D projection, and publishes Detection3DArray and MarkerArray.

    Does NOT run YOLO inference — pairs with single_stream_detector or any 2D node.
    """

    def __init__(self):
        super().__init__('depth_fusion_node')

        # ---------------------------------------------------------------- params
        self.declare_parameter('detections_2d_topic', 'yolo/detection_results')
        self.declare_parameter('rgb_camera_info_topic', 'carla/ego_vehicle/rgb_front/camera_info')
        self.declare_parameter('depth_image_topic', 'carla/ego_vehicle/depth_front/image')
        self.declare_parameter('depth_camera_info_topic', 'carla/ego_vehicle/depth_front/camera_info')
        self.declare_parameter('pointcloud_topic', 'carla/ego_vehicle/lidar')
        self.declare_parameter('qos', 'SENSOR_DATA')
        self.declare_parameter('queue_size', 1)
        self.declare_parameter(
            'synchronization_interval', 0.1,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='0: exact sync; >0: approximate sync slop in seconds.',
            ))
        self.declare_parameter('use_gpu', True)
        self.declare_parameter('use_depth', True)
        self.declare_parameter('use_pointcloud', False)
        self.declare_parameter('output_frame', 'ego_vehicle')
        self.declare_parameter(
            'depth_scale', 1.0,
            ParameterDescriptor(
                description='Divide raw depth pixel values by this to get metres. '
                            '1.0 for Carla 32FC1; 1000.0 for RealSense 16UC1.',
            ))
        self.declare_parameter('depth_max', 50.0)
        self.declare_parameter('transform_timeout', 2.0)
        self.declare_parameter('static_camera_to_robot_tf', True)
        self.declare_parameter('static_camera_info', True)
        self.declare_parameter('cluster_tolerance', 1.0)
        self.declare_parameter('min_cluster_size', 5)
        self.declare_parameter('max_cluster_size', 1000)
        self.declare_parameter('cluster_min_height', 0.1)
        self.declare_parameter('cluster_max_height', 2.0)
        self.declare_parameter('bounding_box_type', 'AABB')

        gp = self.get_parameter
        self.detections_2d_topic = gp('detections_2d_topic').value
        self.rgb_camera_info_topic = gp('rgb_camera_info_topic').value
        self.depth_image_topic = gp('depth_image_topic').value
        self.depth_camera_info_topic = gp('depth_camera_info_topic').value
        self.pointcloud_topic = gp('pointcloud_topic').value
        self.qos = gp('qos').value
        self.queue_size = gp('queue_size').value
        self.synchronization_interval = gp('synchronization_interval').value
        self.use_gpu = gp('use_gpu').value
        self.use_depth = gp('use_depth').value
        self.use_pointcloud = gp('use_pointcloud').value
        self.output_frame = gp('output_frame').value
        self.depth_scale = gp('depth_scale').value
        self.depth_max = gp('depth_max').value
        self.transform_timeout = gp('transform_timeout').value
        self.static_camera_to_robot_tf = gp('static_camera_to_robot_tf').value
        self.static_camera_info = gp('static_camera_info').value
        self.cluster_tolerance = gp('cluster_tolerance').value
        self.min_cluster_size = gp('min_cluster_size').value
        self.max_cluster_size = gp('max_cluster_size').value
        self.cluster_min_height = gp('cluster_min_height').value
        self.cluster_max_height = gp('cluster_max_height').value
        self.bounding_box_type = gp('bounding_box_type').value

        # ---------------------------------------------------------------- device
        self._setup_device()

        # ---------------------------------------------------------------- camera models / state
        self.bridge = CvBridge()
        self.rgb_camera_model = PinholeCameraModel()
        self.depth_camera_model = PinholeCameraModel()
        self.rgb_camera_info = None
        self.depth_camera_info = None
        self.rgb_frame_id = None
        self.depth_frame_id = None
        self.pc_frame_id = None

        # ---------------------------------------------------------------- TF state
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.camera_to_robot_tf = None          # pc/lidar → output_frame
        self.camera_to_robot_tf_o3d = None
        # depth → RGB alignment transform (cached separately from pc → RGB)
        self._depth_to_rgb_tf = None
        self._depth_to_rgb_tf_torch = None
        # pc → RGB projection transform
        self._pc_to_rgb_tf = None
        self._pc_to_rgb_tf_o3d = None

        # ---------------------------------------------------------------- Open3D
        if self.use_pointcloud:
            if not OPEN3D_AVAILABLE:
                self.get_logger().warn('Open3D not available — pointcloud fusion disabled.')
                self.use_pointcloud = False
            else:
                gpu_ok = (self.use_gpu and torch.cuda.is_available()
                          and o3d.core.cuda.is_available())
                self.o3d_device = o3d.core.Device('CUDA:0' if gpu_ok else 'CPU:0')
                self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)

        qos_profile = self._build_qos_profile()

        # ---------------------------------------------------------------- synchronized subscriptions
        self._subs = []
        self._depth_idx = None
        self._pc_idx = None

        self._det_sub = Subscriber(
            self, Detection2DArray, self.detections_2d_topic, qos_profile=qos_profile)
        self._subs.append(self._det_sub)

        if self.use_depth:
            self._depth_sub = Subscriber(
                self, Image, self.depth_image_topic, qos_profile=qos_profile)
            self._depth_idx = len(self._subs)
            self._subs.append(self._depth_sub)

        if self.use_pointcloud:
            self._pc_sub = Subscriber(
                self, PointCloud2, self.pointcloud_topic, qos_profile=qos_profile)
            self._pc_idx = len(self._subs)
            self._subs.append(self._pc_sub)

        if len(self._subs) > 1:
            if self.synchronization_interval > 0.0:
                self._ts = ApproximateTimeSynchronizer(
                    self._subs, self.queue_size, slop=self.synchronization_interval)
            else:
                self._ts = TimeSynchronizer(self._subs, self.queue_size)
            self._ts.registerCallback(self.fusion_callback)
        else:
            self.get_logger().warn(
                'depth_fusion_node: use_depth and use_pointcloud are both False — no 3D output.')

        # camera infos are not in the sync group (low-frequency, effectively latched)
        self.create_subscription(CameraInfo, self.rgb_camera_info_topic,
                                  self._rgb_camera_info_cb, qos_profile)
        if self.use_depth:
            self.create_subscription(CameraInfo, self.depth_camera_info_topic,
                                      self._depth_camera_info_cb, qos_profile)

        # ---------------------------------------------------------------- publishers
        if self.use_depth:
            self.detection3d_depth_pub = self.create_publisher(
                Detection3DArray, 'depth_fusion/detection3d_depth', qos_profile)
            self.marker_depth_pub = self.create_publisher(
                MarkerArray, 'depth_fusion/markers_depth', qos_profile)
        if self.use_pointcloud:
            self.detection3d_pc_pub = self.create_publisher(
                Detection3DArray, 'depth_fusion/detection3d_pointcloud', qos_profile)
            self.marker_pc_pub = self.create_publisher(
                MarkerArray, 'depth_fusion/markers_pointcloud', qos_profile)

        self.get_logger().info(
            f'depth_fusion_node started. '
            f'depth={self.use_depth}, pointcloud={self.use_pointcloud}, '
            f'output_frame="{self.output_frame}"'
        )

    # ------------------------------------------------------------ camera info callbacks

    def _rgb_camera_info_cb(self, msg):
        if self.rgb_camera_info is None or not self.static_camera_info:
            self.rgb_camera_info = msg
            self.rgb_camera_model.fromCameraInfo(msg)
            self.rgb_frame_id = msg.header.frame_id

    def _depth_camera_info_cb(self, msg):
        if self.depth_camera_info is None or not self.static_camera_info:
            self.depth_camera_info = msg
            self.depth_camera_model.fromCameraInfo(msg)
            self.depth_frame_id = msg.header.frame_id

    # ------------------------------------------------------------ main callback

    def fusion_callback(self, *msgs):
        if self.rgb_camera_info is None:
            self.get_logger().warn('Waiting for RGB CameraInfo...', once=True)
            return

        detections_msg = msgs[0]
        if not detections_msg.detections:
            return

        depth_msg = msgs[self._depth_idx] if self._depth_idx is not None else None
        pc_msg = msgs[self._pc_idx] if self._pc_idx is not None else None

        if depth_msg is not None and self.depth_camera_info is not None:
            try:
                self._process_depth(detections_msg, depth_msg)
            except Exception as e:
                self.get_logger().error(f'Depth fusion error: {e}')

        if pc_msg is not None:
            try:
                self._process_pointcloud(detections_msg, pc_msg)
            except Exception as e:
                self.get_logger().error(f'Pointcloud fusion error: {e}')

    # ------------------------------------------------------------ depth path

    def _process_depth(self, detections_msg, depth_msg):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth image decode failed: {e}')
            return
        depth_image = depth_image.astype(np.float32)
        self.depth_frame_id = depth_msg.header.frame_id

        # optionally warp depth image to RGB frame (once if static)
        self._update_depth_to_rgb_tf()
        if (self._depth_to_rgb_tf_torch is not None
                and not torch.equal(
                    self._depth_to_rgb_tf_torch,
                    torch.eye(4, dtype=torch.float32, device=self.torch_device))):
            depth_image = self._align_depth_to_rgb(
                torch.from_numpy(depth_image).to(dtype=torch.float32, device=self.torch_device),
                torch.from_numpy(self.depth_camera_model.K).to(dtype=torch.float32, device=self.torch_device),
                torch.from_numpy(self.rgb_camera_model.K).to(dtype=torch.float32, device=self.torch_device),
                self._depth_to_rgb_tf_torch,
                (self.rgb_camera_model.height, self.rgb_camera_model.width),
                self.depth_scale,
            ).cpu().numpy()

        frame_id = self.output_frame or self.depth_frame_id or ''
        timestamp = depth_msg.header.stamp
        det3d_arr = Detection3DArray()
        det3d_arr.header.frame_id = frame_id
        det3d_arr.header.stamp = timestamp
        marker_arr = MarkerArray()

        for i, det in enumerate(detections_msg.detections):
            bbox = (det.bbox.center.position.x, det.bbox.center.position.y,
                    det.bbox.size_x, det.bbox.size_y)
            conf = det.results[0].hypothesis.score if det.results else 0.0
            cls_name = det.results[0].hypothesis.class_id if det.results else ''

            result = self._project_depth(bbox, depth_image)
            if result is None:
                continue
            x3, y3, z3, sx, sy, sz = result

            det3d_arr.detections.append(
                self._make_detection3d(x3, y3, z3, sx, sy, sz, conf, cls_name, frame_id))
            marker_arr.markers.append(
                create_marker(i, x3, y3, z3, sx, sy, sz, frame_id,
                              timestamp=timestamp, rgba=[1.0, 0.0, 0.0, 0.5]))

        self.detection3d_depth_pub.publish(det3d_arr)
        self.marker_depth_pub.publish(marker_arr)

    def _update_depth_to_rgb_tf(self):
        if self._depth_to_rgb_tf is not None and self.static_camera_to_robot_tf:
            return
        if not (self.depth_frame_id and self.rgb_frame_id):
            return
        transform = self.lookup_transform(self.depth_frame_id, self.rgb_frame_id,
                                           rclpy.time.Time())
        if transform is not None:
            self._depth_to_rgb_tf = self._transform_to_matrix(transform)
            self._depth_to_rgb_tf_torch = torch.as_tensor(
                self._depth_to_rgb_tf, dtype=torch.float32, device=self.torch_device)

    def _project_depth(self, xywh, depth_image):
        """Back-project a 2D bbox into 3D using a depth image.

        Returns (x, y, z, size_x, size_y, size_z) in depth camera frame, or None.
        """
        cx_b, cy_b = int(xywh[0]), int(xywh[1])
        w_b, h_b = int(xywh[2]), int(xywh[3])
        H, W = depth_image.shape[:2]

        u0 = max(cx_b - w_b // 2, 0)
        u1 = min(cx_b + w_b // 2, W - 1)
        v0 = max(cy_b - h_b // 2, 0)
        v1 = min(cy_b + h_b // 2, H - 1)
        if u1 <= u0 or v1 <= v0:
            return None

        roi = depth_image[v0:v1, u0:u1] / self.depth_scale
        if not np.any(roi):
            return None

        # centre pixel depth as reference, filter by depth_max
        cx_b_c = max(0, min(cx_b, W - 1))
        cy_b_c = max(0, min(cy_b, H - 1))
        z_ref = float(depth_image[cy_b_c, cx_b_c]) / self.depth_scale
        mask_z = (roi > 0) & (np.abs(roi - z_ref) <= self.depth_max)
        if not np.any(mask_z):
            return None

        roi_filt = roi[mask_z]
        z_min_v, z_max_v = float(np.min(roi_filt)), float(np.max(roi_filt))
        z = (z_min_v + z_max_v) / 2.0
        if z == 0.0:
            return None

        dcx = self.depth_camera_model.cx()
        dcy = self.depth_camera_model.cy()
        dfx = self.depth_camera_model.fx()
        dfy = self.depth_camera_model.fy()

        x = z * (cx_b - dcx) / dfx
        y = z * (cy_b - dcy) / dfy
        size_x = z * (w_b / dfx)
        size_y = z * (h_b / dfy)
        size_z = float(z_max_v - z_min_v)
        return x, y, z, size_x, size_y, size_z

    @staticmethod
    def _align_depth_to_rgb(depth_map, K_depth, K_rgb, T_depth_to_rgb, rgb_shape, depth_scale=1.0):
        """Warp a depth image from the depth camera frame to the RGB camera frame."""
        H_d, W_d = depth_map.shape
        H_rgb, W_rgb = rgb_shape
        device = depth_map.device

        v, u = torch.meshgrid(torch.arange(H_d, device=device),
                               torch.arange(W_d, device=device), indexing='ij')
        u, v, z = u.flatten(), v.flatten(), depth_map.flatten() / depth_scale
        valid = z > 0
        u, v, z = u[valid], v[valid], z[valid]

        cx_d, cy_d = K_depth[0, 2], K_depth[1, 2]
        fx_d, fy_d = K_depth[0, 0], K_depth[1, 1]
        pts = torch.stack([(u - cx_d) * z / fx_d,
                            (v - cy_d) * z / fy_d,
                            z,
                            torch.ones_like(z)], dim=0)  # 4×N
        pts_r = T_depth_to_rgb @ pts

        x_r, y_r, z_r = pts_r[0], pts_r[1], pts_r[2]
        ok = z_r > 0
        x_r, y_r, z_r = x_r[ok], y_r[ok], z_r[ok]

        cx_r, cy_r = K_rgb[0, 2], K_rgb[1, 2]
        fx_r, fy_r = K_rgb[0, 0], K_rgb[1, 1]
        u_p = torch.round((x_r * fx_r / z_r) + cx_r).long()
        v_p = torch.round((y_r * fy_r / z_r) + cy_r).long()

        inb = (u_p >= 0) & (u_p < W_rgb) & (v_p >= 0) & (v_p < H_rgb)
        u_f, v_f, z_f = u_p[inb], v_p[inb], z_r[inb]

        # painter's algorithm: farther points first so nearer points overwrite
        order = torch.argsort(z_f, descending=True)
        u_f, v_f, z_f = u_f[order], v_f[order], z_f[order]

        out = torch.zeros((H_rgb, W_rgb), device=device, dtype=torch.float32)
        out[v_f, u_f] = z_f
        return out * depth_scale

    # ------------------------------------------------------------ pointcloud path

    def _process_pointcloud(self, detections_msg, pc_msg):
        self.pc_frame_id = pc_msg.header.frame_id

        # TF: pc frame → output_frame (for world-space results)
        self._update_camera_to_robot_tf(pc_msg.header.stamp)
        # TF: pc/output frame → RGB frame (for 2D projection)
        self._update_pc_to_rgb_tf()

        _, _, points_np = unpack_pointcloud_message(pc_msg, fields=('x', 'y', 'z'))
        if points_np is None or len(points_np) == 0:
            return

        self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)
        self.o3d_pointcloud.point.positions = o3c.Tensor(
            points_np, dtype=o3c.Dtype.Float32, device=self.o3d_device)
        if self.o3d_pointcloud.is_empty():
            return

        frame_id = self.output_frame or self.pc_frame_id or ''
        timestamp = pc_msg.header.stamp
        det3d_arr = Detection3DArray()
        det3d_arr.header.frame_id = frame_id
        det3d_arr.header.stamp = timestamp
        marker_arr = MarkerArray()

        for i, det in enumerate(detections_msg.detections):
            bbox = (det.bbox.center.position.x, det.bbox.center.position.y,
                    det.bbox.size_x, det.bbox.size_y)
            conf = det.results[0].hypothesis.score if det.results else 0.0
            cls_name = det.results[0].hypothesis.class_id if det.results else ''

            result = self._project_pointcloud(bbox)
            if result is None:
                continue
            x3, y3, z3, sx, sy, sz, quat = result

            det3d_arr.detections.append(
                self._make_detection3d(x3, y3, z3, sx, sy, sz, conf, cls_name,
                                        quat=quat, frame_id=frame_id))
            marker_arr.markers.append(
                create_marker(i, x3, y3, z3, sx, sy, sz, frame_id,
                              timestamp=timestamp, quat=quat, rgba=[0.0, 1.0, 0.0, 0.5]))

        self.detection3d_pc_pub.publish(det3d_arr)
        self.marker_pc_pub.publish(marker_arr)

    def _update_camera_to_robot_tf(self, timestamp=None):
        """Cache the pointcloud-frame → output_frame transform."""
        if self.camera_to_robot_tf is not None and self.static_camera_to_robot_tf:
            return
        if not (self.output_frame and self.pc_frame_id):
            return
        t = self.lookup_transform(self.pc_frame_id, self.output_frame, timestamp)
        if t is not None:
            self.camera_to_robot_tf = self._transform_to_matrix(t)
            if OPEN3D_AVAILABLE:
                self.camera_to_robot_tf_o3d = o3c.Tensor(
                    self.camera_to_robot_tf, dtype=o3c.float32, device=self.o3d_device)

    def _update_pc_to_rgb_tf(self):
        """Cache the source-frame → RGB camera frame transform for 2D projection."""
        if self._pc_to_rgb_tf is not None and self.static_camera_to_robot_tf:
            return
        if not self.rgb_frame_id:
            return
        # project from the output frame if we've already transformed the cloud, else from pc frame
        src = (self.output_frame
               if (self.output_frame and self.camera_to_robot_tf is not None)
               else self.pc_frame_id)
        if not src:
            return
        t = self.lookup_transform(src, self.rgb_frame_id, rclpy.time.Time())
        if t is not None:
            self._pc_to_rgb_tf = self._transform_to_matrix(t)
            self._pc_to_rgb_tf_o3d = o3c.Tensor(
                self._pc_to_rgb_tf, dtype=o3c.float32, device=self.o3d_device)

    def _project_pointcloud(self, xywh):
        """Filter 3D points inside a 2D bbox, cluster, and return the dominant cluster.

        Returns (x, y, z, size_x, size_y, size_z, quat) or None.
        """
        if not self.rgb_camera_model.width or not self.rgb_camera_model.height:
            return None

        bbox_cx, bbox_cy = int(xywh[0]), int(xywh[1])
        bbox_w, bbox_h = int(xywh[2]), int(xywh[3])
        rcx = self.rgb_camera_model.cx()
        rcy = self.rgb_camera_model.cy()
        rfx = self.rgb_camera_model.fx()
        rfy = self.rgb_camera_model.fy()
        img_w = self.rgb_camera_model.width
        img_h = self.rgb_camera_model.height

        # transform pointcloud to RGB camera frame for 2D projection
        if self._pc_to_rgb_tf_o3d is not None:
            pts = self.o3d_pointcloud.clone().transform(self._pc_to_rgb_tf_o3d).point.positions
        else:
            pts = self.o3d_pointcloud.point.positions.clone()

        x_ = pts[:, 0]
        y_ = pts[:, 1]
        z_ = pts[:, 2]

        # project to 2D
        x_2d = (x_ * rfx / z_) + rcx
        y_2d = (y_ * rfy / z_) + rcy

        # filter: within image FOV, within bbox, in front of camera
        x1 = bbox_cx - bbox_w // 2
        x2 = bbox_cx + bbox_w // 2
        y1 = bbox_cy - bbox_h // 2
        y2 = bbox_cy + bbox_h // 2
        mask = ((z_ > 0) &
                (x_2d >= 0) & (x_2d < img_w) &
                (y_2d >= 0) & (y_2d < img_h) &
                (x_2d >= x1) & (x_2d <= x2) &
                (y_2d >= y1) & (y_2d <= y2))

        # select the matching points from the ORIGINAL (output-frame) pointcloud
        in_bbox = self.o3d_pointcloud.select_by_mask(mask)
        if in_bbox.is_empty():
            return None

        clusters, bboxes, centers, extents, quats = self._cluster_points(in_bbox)
        if not bboxes:
            return None
        x3, y3, z3 = centers[0]
        sx, sy, sz = extents[0]
        return x3, y3, z3, sx, sy, sz, quats[0]

    def _cluster_points(self, o3d_pcd):
        """DBSCAN cluster an Open3D tensor PointCloud.

        Returns (clusters, bboxes, centers, extents, quats).
        Results are sorted by cluster size (largest first).
        """
        if o3d_pcd.is_empty():
            return [], [], [], [], []

        labels = o3d_pcd.cluster_dbscan(
            eps=self.cluster_tolerance,
            min_points=self.min_cluster_size,
            print_progress=False,
        )
        labels_t = torch.utils.dlpack.from_dlpack(labels.to_dlpack())
        unique, counts = torch.unique(labels_t[labels_t != -1], return_counts=True)
        if len(unique) == 0:
            return [], [], [], [], []

        # sort by cluster size descending so callers get the largest cluster first
        order = torch.argsort(counts, descending=True)
        unique = unique[order]

        clusters, bboxes, centers, extents, quats = [], [], [], [], []
        for label in unique:
            mask_bool = (labels_t == label).to(dtype=torch.uint8).contiguous()
            mask_o3d = o3c.Tensor.from_dlpack(
                torch.utils.dlpack.to_dlpack(mask_bool)).to(o3c.Dtype.Bool)
            cluster = o3d_pcd.select_by_mask(mask_o3d)
            if cluster.is_empty():
                continue

            pts = cluster.point.positions
            min_z = pts[:, 2].min().item()
            max_z = pts[:, 2].max().item()
            height = max_z - min_z
            if not (self.cluster_min_height <= height <= self.cluster_max_height):
                continue

            use_obb = (self.bounding_box_type.lower() == 'obb' and _TF_TRANSFORMS)
            if use_obb:
                bb = cluster.get_oriented_bounding_box()
                center = bb.center.cpu().numpy().tolist()
                extent = bb.extent.cpu().numpy().tolist()
                R_mat = bb.rotation.cpu().numpy()
                R4 = np.vstack((np.hstack((R_mat, [[0], [0], [0]])), [0, 0, 0, 1]))
                q = quaternion_from_matrix(R4)
                quat = Quaternion(x=float(q[0]), y=float(q[1]),
                                   z=float(q[2]), w=float(q[3]))
            else:
                bb = cluster.get_axis_aligned_bounding_box()
                center = bb.get_center().cpu().numpy().tolist()
                extent = bb.get_extent().cpu().numpy().tolist()
                quat = None

            clusters.append(cluster)
            bboxes.append(bb)
            centers.append(center)
            extents.append(extent)
            quats.append(quat)

        return clusters, bboxes, centers, extents, quats

    # ------------------------------------------------------------ message helpers

    def _make_detection3d(self, x, y, z, sx, sy, sz, confidence, class_id,
                           quat=None, frame_id=None):
        det = Detection3D()
        if frame_id:
            det.header.frame_id = frame_id
        det.bbox.center.position.x = float(x)
        det.bbox.center.position.y = float(y)
        det.bbox.center.position.z = float(z)
        if quat is not None:
            det.bbox.center.orientation = quat
        else:
            det.bbox.center.orientation.w = 1.0
        det.bbox.size.x = max(float(sx), 0.01)
        det.bbox.size.y = max(float(sy), 0.01)
        det.bbox.size.z = max(float(sz), 0.01)
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = str(class_id)
        hyp.hypothesis.score = float(confidence)
        det.results.append(hyp)
        return det

    # ------------------------------------------------------------ TF helpers

    def lookup_transform(self, source_frame, target_frame, timestamp=None):
        if timestamp is None:
            timestamp = rclpy.time.Time()
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, timestamp,
                rclpy.duration.Duration(seconds=self.transform_timeout))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup {source_frame}→{target_frame} failed: {e}',
                                    once=True)
            return None

    def _transform_to_matrix(self, transform):
        """Convert a geometry_msgs/TransformStamped to a 4×4 numpy matrix."""
        t = transform.transform.translation
        r = transform.transform.rotation
        q = [r.x, r.y, r.z, r.w]
        mat = quaternion_matrix(q) if _TF_TRANSFORMS else _quaternion_to_matrix(q)
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    # ------------------------------------------------------------ cleanup

    def destroy_node(self):
        if OPEN3D_AVAILABLE and self.use_pointcloud and hasattr(self, 'o3d_pointcloud'):
            self.o3d_pointcloud.clear()
            del self.o3d_pointcloud
        if self.use_gpu and 'cuda' in self.device and torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DepthFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()