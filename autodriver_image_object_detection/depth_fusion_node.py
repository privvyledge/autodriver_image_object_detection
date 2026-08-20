"""Depth fusion node: lift Detection2DArray to 3D using depth images or LiDAR pointclouds."""
import numpy as np
import torch

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
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
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

from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

from autodriver_image_object_detection.base_detector import BaseDetector
from autodriver_image_object_detection.utils.common import make_deleteall_marker_array
from autodriver_image_object_detection.utils.pointcloud_utils import (
    unpack_pointcloud_message, create_marker, project_depth_to_3d,
)
from autodriver_image_object_detection.utils.profiling import setup_profiler, apply_profiler_param


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

    # Floor for the measured depth-path view-axis box extent so a flat/degenerate
    # depth spread still yields a visible (non-zero) box.
    MIN_DEPTH_THICKNESS = 0.2

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
        self.declare_parameter('fps', 30)
        _fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.declare_parameter(
            'synchronization_interval', 1.5 / _fps,
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
        self.declare_parameter(
            'optical_frame_id', '',
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='TF frame that carries the camera OPTICAL convention (x-right, y-down, '
                            'z-forward) — the frame the pinhole projection actually produces points '
                            'in. Leave EMPTY (default) when camera_info.header.frame_id is already '
                            'the optical frame: the REP-103 norm for RealSense '
                            '(camera_color_optical_frame), ZED (zed_*_camera_optical_frame), gscam, '
                            'AND carla-ros-bridge (verified: ego_vehicle->rgb_front is RPY '
                            '[-90,0,-90] = X-right,Y-down,Z-forward; CARLA bakes the optical '
                            'orientation into the camera frame rather than shipping a separate '
                            '_optical child). The node then projects, stamps with that frame and '
                            'lets TF do all rotation — no manual axis math, no static transform. '
                            'Set this ONLY for a non-compliant driver whose camera_info frame is a '
                            'BODY frame (x-fwd,y-left,z-up) with no optical child: publish a static '
                            'optical child (body->optical = RPY [-90,0,-90]) and point this at it.'))
        self.declare_parameter(
            'depth_box_thickness', 4.0,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='Depth path only. UPPER BOUND (m) on the view-axis box extent. The '
                            'extent is measured from the ROI depth spread (10-90th pct); this caps '
                            'it (~a large vehicle length) so background bleed in the bbox-only ROI '
                            "can't make an absurd box. Floored at MIN_DEPTH_THICKNESS."))
        self.declare_parameter(
            'publish_empty_detections', True,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='On a zero-detection frame, skip projection but still publish an '
                            'empty Detection3DArray + DELETEALL MarkerArray (heartbeat for '
                            'downstream costmap/obstacle clearing). False restores the original '
                            'behaviour of publishing nothing on empty frames.',
            ))

        gp = self.get_parameter
        self.detections_2d_topic = gp('detections_2d_topic').value
        self.rgb_camera_info_topic = gp('rgb_camera_info_topic').value
        self.depth_image_topic = gp('depth_image_topic').value
        self.depth_camera_info_topic = gp('depth_camera_info_topic').value
        self.pointcloud_topic = gp('pointcloud_topic').value
        self.qos = gp('qos').value
        self.queue_size = gp('queue_size').value
        self.fps = gp('fps').value
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
        self.optical_frame_id = gp('optical_frame_id').value
        self.depth_box_thickness = gp('depth_box_thickness').value
        self.publish_empty_detections = gp('publish_empty_detections').value

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
        # RGB camera frame → output_frame, used to place depth-projected 3D points
        self._optical_to_output_tf = None
        # pc → RGB projection transform
        self._pc_to_optical_tf = None
        self._pc_to_optical_tf_o3d = None
        self._pc_to_optical_tf_src = None   # source frame the cached pc→rgb tf was built for

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
        # BEST_EFFORT with queue_size depth so the synchronizer has enough history to match.
        # _build_sensor_qos_profile() uses depth=1 which is too aggressive for synced topics.
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.queue_size,
        )

        # ---------------------------------------------------------------- synchronized subscriptions
        # Two independent 2-way synchronizers so the depth path and pointcloud path fire
        # independently. A single 3-way synchronizer would require all three topics to match
        # simultaneously — if pointcloud timestamps drift, depth output silently stops too.
        _has_output = False

        if self.use_depth:
            self._det_depth_sub = Subscriber(
                self, Detection2DArray, self.detections_2d_topic, qos_profile=qos_profile,
                callback_group=self._sub_cb_group)
            self._depth_sub = Subscriber(
                self, Image, self.depth_image_topic, qos_profile=sensor_qos,
                callback_group=self._sub_cb_group)
            if self.synchronization_interval > 0.0:
                self._depth_ts = ApproximateTimeSynchronizer(
                    [self._det_depth_sub, self._depth_sub],
                    self.queue_size, slop=self.synchronization_interval)
            else:
                self._depth_ts = TimeSynchronizer(
                    [self._det_depth_sub, self._depth_sub], self.queue_size)
            self._depth_ts.registerCallback(self._depth_fusion_callback)
            _has_output = True

        if self.use_pointcloud:
            self._det_pc_sub = Subscriber(
                self, Detection2DArray, self.detections_2d_topic, qos_profile=qos_profile,
                callback_group=self._sub_cb_group)
            self._pc_sub = Subscriber(
                self, PointCloud2, self.pointcloud_topic, qos_profile=sensor_qos,
                callback_group=self._sub_cb_group)
            if self.synchronization_interval > 0.0:
                self._pc_ts = ApproximateTimeSynchronizer(
                    [self._det_pc_sub, self._pc_sub],
                    self.queue_size, slop=self.synchronization_interval)
            else:
                self._pc_ts = TimeSynchronizer(
                    [self._det_pc_sub, self._pc_sub], self.queue_size)
            self._pc_ts.registerCallback(self._pc_fusion_callback)
            _has_output = True

        if not _has_output:
            self.get_logger().warn(
                'depth_fusion_node: use_depth and use_pointcloud are both False — no 3D output.')

        # camera infos are not in the sync group (low-frequency, effectively latched)
        self.create_subscription(CameraInfo, self.rgb_camera_info_topic,
                                  self._rgb_camera_info_cb, qos_profile,
                                  callback_group=self._sub_cb_group)
        if self.use_depth:
            self.create_subscription(CameraInfo, self.depth_camera_info_topic,
                                      self._depth_camera_info_cb, qos_profile,
                                      callback_group=self._sub_cb_group)

        # ---------------------------------------------------------------- publishers
        if self.use_depth:
            self.detection3d_depth_pub = self.create_publisher(
                Detection3DArray, 'depth_fusion/detection3d_depth', self.queue_size)
            self.marker_depth_pub = self.create_publisher(
                MarkerArray, 'depth_fusion/markers_depth', self.queue_size)
        if self.use_pointcloud:
            self.detection3d_pc_pub = self.create_publisher(
                Detection3DArray, 'depth_fusion/detection3d_pointcloud', self.queue_size)
            self.marker_pc_pub = self.create_publisher(
                MarkerArray, 'depth_fusion/markers_pointcloud', self.queue_size)

        self.profiler = setup_profiler(self)
        self.add_on_set_parameters_callback(self.parameter_change_callback)

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

    # ------------------------------------------------------------ main callbacks

    def _depth_fusion_callback(self, detections_msg, depth_msg):
        if self.rgb_camera_info is None:
            self.get_logger().warn('Waiting for RGB CameraInfo...', once=True)
            return
        if not detections_msg.detections:
            # Zero detections: skip the expensive projection but emit a heartbeat so
            # downstream costmaps can clear and stale RViz markers are removed.
            self._publish_empty_3d(self.detection3d_depth_pub, self.marker_depth_pub,
                                    detections_msg.header)
            return
        if self.depth_camera_info is None:
            self.get_logger().warn('Waiting for depth CameraInfo...', once=True)
            return
        try:
            with self.profiler.measure("depth_projection"):
                self._process_depth(detections_msg, depth_msg)
        except Exception as e:
            self.get_logger().error(f'Depth fusion error: {e}')
        self.profiler.flush(self)

    def _pc_fusion_callback(self, detections_msg, pc_msg):
        if self.rgb_camera_info is None:
            self.get_logger().warn('Waiting for RGB CameraInfo...', once=True)
            return
        if not detections_msg.detections:
            self._publish_empty_3d(self.detection3d_pc_pub, self.marker_pc_pub,
                                    detections_msg.header)
            return
        try:
            with self.profiler.measure("pointcloud_projection"):
                self._process_pointcloud(detections_msg, pc_msg)
        except Exception as e:
            self.get_logger().error(f'Pointcloud fusion error: {e}')
        self.profiler.flush(self)

    def _publish_empty_3d(self, det_pub, marker_pub, header):
        """Publish an empty Detection3DArray + DELETEALL markers (heartbeat)."""
        if not self.publish_empty_detections:
            return
        det3d_arr = Detection3DArray()
        det3d_arr.header.frame_id = self.output_frame or header.frame_id or ''
        det3d_arr.header.stamp = header.stamp
        det_pub.publish(det3d_arr)
        marker_pub.publish(make_deleteall_marker_array())

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

        timestamp = depth_msg.header.stamp
        depth_image[depth_image > self.depth_max * self.depth_scale] = 0.0
        rfx = self.rgb_camera_model.fx()
        rfy = self.rgb_camera_model.fy()

        # The pinhole projection yields points in the camera OPTICAL frame; TF carries the
        # optical->output rotation in one shot (no manual axis swaps). If the TF isn't
        # available yet, stamp honestly with the optical frame rather than mislabelling the
        # optical-axis coords as output_frame.
        self._update_optical_to_output_tf()
        T = self._optical_to_output_tf
        frame_id = (self.output_frame if (T is not None and self.output_frame)
                    else self._optical_frame()) or ''
        det3d_arr = Detection3DArray()
        det3d_arr.header.frame_id = frame_id
        det3d_arr.header.stamp = timestamp
        marker_arr = MarkerArray()

        for i, det in enumerate(detections_msg.detections):
            self._warn_if_oriented(det)
            bbox = (det.bbox.center.position.x, det.bbox.center.position.y,
                    det.bbox.size_x, det.bbox.size_y)
            conf = det.results[0].hypothesis.score if det.results else 0.0
            cls_name = det.results[0].hypothesis.class_id if det.results else ''

            xyz = project_depth_to_3d(bbox, depth_image, self.rgb_camera_model, self.depth_scale)
            if xyz is None:
                continue
            # project_depth_to_3d returns the point in camera-OPTICAL axes
            # (x-right, y-down, z-forward), the centroid range z_o, and z_ext = the
            # view-axis depth spread. Lateral/vertical extents come from the image-plane
            # bbox at range z_o.
            x_o, y_o, z_o, z_ext = xyz
            width = z_o * int(bbox[2]) / rfx       # image-plane width  (metres)
            height = z_o * int(bbox[3]) / rfy      # image-plane height (metres)

            # Point is in optical axes; TF rotates optical->output in one shot.
            px, py, pz = x_o, y_o, z_o
            if T is not None:
                p = T @ np.array([px, py, pz, 1.0], dtype=np.float64)
                px, py, pz = float(p[0]), float(p[1]), float(p[2])

            # View-axis (depth) extent is MEASURED from the depth spread, floored so the
            # box stays visible and capped by depth_box_thickness — an upper bound (~a
            # large vehicle's length) so background bleed in the bbox-only ROI can't
            # produce an absurd box. A mask path (future) would tighten z_ext directly.
            thickness = min(max(z_ext, self.MIN_DEPTH_THICKNESS), self.depth_box_thickness)

            # Box extents assume a body/world output_frame (x-fwd, y-left, z-up):
            # thickness along view axis, width lateral, height vertical.
            sx = thickness
            sy = width
            sz = height

            det3d_arr.detections.append(
                self._make_detection3d(px, py, pz, sx, sy, sz, conf, cls_name, frame_id=frame_id))
            marker_arr.markers.append(
                create_marker(i, px, py, pz, sx, sy, sz, frame_id,
                              timestamp=timestamp, rgba=[1.0, 0.0, 0.0, 0.5]))

        self.detection3d_depth_pub.publish(det3d_arr)
        self.marker_depth_pub.publish(marker_arr)

    def _optical_frame(self):
        """TF frame the pinhole projection produces points in.

        Empty optical_frame_id => camera_info.header.frame_id is itself the optical
        frame (REP-103 norm: RealSense/ZED/gscam). Otherwise the explicit optical child
        the user published for a non-optical driver (e.g. carla-ros-bridge)."""
        return self.optical_frame_id or self.rgb_frame_id

    def _update_optical_to_output_tf(self):
        """Cache the optical-frame → output_frame transform (depth 3D placement)."""
        if self._optical_to_output_tf is not None and self.static_camera_to_robot_tf:
            return
        src = self._optical_frame()
        if not (src and self.output_frame):
            return
        if src == self.output_frame:
            self._optical_to_output_tf = np.eye(4, dtype=np.float64)
            return
        t = self.lookup_transform(src, self.output_frame, rclpy.time.Time())
        if t is not None:
            self._optical_to_output_tf = self._transform_to_matrix(t)

    def _update_depth_to_rgb_tf(self):
        if self._depth_to_rgb_tf is not None and self.static_camera_to_robot_tf:
            return
        if not (self.depth_frame_id and self.rgb_frame_id):
            return
        if self.depth_frame_id == self.rgb_frame_id:
            self._depth_to_rgb_tf = np.eye(4, dtype=np.float64)
            self._depth_to_rgb_tf_torch = torch.eye(
                4, dtype=torch.float32, device=self.torch_device)
            return
        transform = self.lookup_transform(self.depth_frame_id, self.rgb_frame_id,
                                           rclpy.time.Time())
        if transform is not None:
            self._depth_to_rgb_tf = self._transform_to_matrix(transform)
            self._depth_to_rgb_tf_torch = torch.as_tensor(
                self._depth_to_rgb_tf, dtype=torch.float32, device=self.torch_device)

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
        self._update_pc_to_optical_tf()

        _, _, points_np = unpack_pointcloud_message(pc_msg, fields=('x', 'y', 'z'))
        if points_np is None or len(points_np) == 0:
            return

        self.o3d_pointcloud = o3d.t.geometry.PointCloud(self.o3d_device)
        self.o3d_pointcloud.point.positions = o3c.Tensor(
            points_np, dtype=o3c.Dtype.Float32, device=self.o3d_device)
        if self.o3d_pointcloud.is_empty():
            return

        # Move the cloud into output_frame BEFORE clustering so the resulting 3D centroids
        # are actually in output_frame (the header claims it). Without this the clusters
        # stay in the raw lidar frame and the published positions are offset by the
        # lidar→robot mounting transform. _update_pc_to_optical_tf() already projects from
        # output_frame when camera_to_robot_tf is available, so the 2D filter stays valid.
        if (self.camera_to_robot_tf_o3d is not None
                and not np.allclose(self.camera_to_robot_tf, np.eye(4))):
            self.o3d_pointcloud = self.o3d_pointcloud.transform(self.camera_to_robot_tf_o3d)

        frame_id = self.output_frame or self.pc_frame_id or ''
        timestamp = pc_msg.header.stamp
        det3d_arr = Detection3DArray()
        det3d_arr.header.frame_id = frame_id
        det3d_arr.header.stamp = timestamp
        marker_arr = MarkerArray()

        if (self._pc_to_optical_tf_o3d is not None
                and not np.allclose(self._pc_to_optical_tf, np.eye(4))):
            pts_optical = self.o3d_pointcloud.clone().transform(self._pc_to_optical_tf_o3d).point.positions
        else:
            pts_optical = self.o3d_pointcloud.point.positions

        for i, det in enumerate(detections_msg.detections):
            self._warn_if_oriented(det)
            bbox = (det.bbox.center.position.x, det.bbox.center.position.y,
                    det.bbox.size_x, det.bbox.size_y)
            conf = det.results[0].hypothesis.score if det.results else 0.0
            cls_name = det.results[0].hypothesis.class_id if det.results else ''

            result = self._project_pointcloud(bbox, pts_optical)
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

    def _warn_if_oriented(self, det) -> None:
        """Warn once if an upstream detector is sending rotated 2D boxes.

        Both projection paths here treat size_x/size_y as an axis-aligned ROI.
        A producer running with publish_oriented_bbox writes the mask principal
        axis into bbox.center.theta and the rotated extents into the sizes, so
        the ROI this node cuts would be wrong without any error being raised.
        """
        if getattr(det.bbox.center, 'theta', 0.0) == 0.0:
            return
        if not getattr(self, '_warned_oriented_bbox', False):
            self._warned_oriented_bbox = True
            self.get_logger().warn(
                'Received a Detection2D with a non-zero bbox.center.theta. This node '
                'projects axis-aligned ROIs only, so rotated boxes will cut the wrong '
                'region. Set publish_oriented_bbox:=false on the upstream detector.')

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

    def _update_pc_to_optical_tf(self):
        """Cache the source-frame → camera OPTICAL frame transform for 2D projection.

        Projects the cloud into the optical frame so the pinhole model below sees genuine
        optical axes (x-right, y-down, z-fwd) — no manual swaps. The optical frame is
        camera_info.frame_id (REP-103 norm) or the explicit optical_frame_id."""
        tgt = self._optical_frame()
        if not tgt:
            return
        # project from the output frame if we've already transformed the cloud, else from pc frame
        src = (self.output_frame
               if (self.output_frame and self.camera_to_robot_tf is not None)
               else self.pc_frame_id)
        if not src:
            return
        # Recompute when the source frame changes, even under static caching. At startup
        # the camera→robot TF may not resolve on the first callback, so src is the raw pc
        # frame; once it resolves the cloud is transformed into output_frame and src flips.
        # A stale pc→optical transform then projects the wrong frame and silently drops
        # every detection — guard on the cached src, not just non-None.
        if (self._pc_to_optical_tf is not None and self.static_camera_to_robot_tf
                and self._pc_to_optical_tf_src == src):
            return
        if src == tgt:
            self._pc_to_optical_tf = np.eye(4, dtype=np.float64)
            if OPEN3D_AVAILABLE:
                self._pc_to_optical_tf_o3d = o3c.Tensor(
                    self._pc_to_optical_tf, dtype=o3c.float32, device=self.o3d_device)
            self._pc_to_optical_tf_src = src
            return
        t = self.lookup_transform(src, tgt, rclpy.time.Time())
        if t is not None:
            self._pc_to_optical_tf = self._transform_to_matrix(t)
            self._pc_to_optical_tf_o3d = o3c.Tensor(
                self._pc_to_optical_tf, dtype=o3c.float32, device=self.o3d_device)
            self._pc_to_optical_tf_src = src

    def _project_pointcloud(self, xywh, pts_optical):
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

        # pts_optical are already in the camera OPTICAL frame (the cloud was TF'd into it),
        # so the pinhole model applies directly: x-right, y-down, z-forward into the scene.
        x_ = pts_optical[:, 0]
        y_ = pts_optical[:, 1]
        z_ = pts_optical[:, 2]

        mask_fwd = (z_ > 0).to(o3c.float32)
        z_denom = z_ * mask_fwd + (1.0 - mask_fwd)
        x_2d = (x_ * rfx / z_denom) + rcx
        y_2d = (y_ * rfy / z_denom) + rcy

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
        labels_np = labels.to(o3c.Device('CPU:0')).numpy()
        valid = labels_np >= 0
        if not valid.any():
            return [], [], [], [], []
        unique_labels, counts = np.unique(labels_np[valid], return_counts=True)
        order = np.argsort(-counts)
        unique_labels = unique_labels[order]

        clusters, bboxes, centers, extents, quats = [], [], [], [], []
        for label in unique_labels:
            indices = np.where(labels_np == label)[0].astype(np.int64)
            cluster = o3d_pcd.select_by_index(
                o3c.Tensor(indices, dtype=o3c.int64, device=self.o3d_device))
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

    # ------------------------------------------------------------ parameter callback

    def parameter_change_callback(self, params):
        """Handle runtime updates for profiling and the empty-3D heartbeat."""
        result = SetParametersResult(successful=True)
        for param in params:
            if apply_profiler_param(self.profiler, param.name, param.value):
                pass
            elif param.name == 'publish_empty_detections' and param.type_ == Parameter.Type.BOOL:
                self.publish_empty_detections = param.value
            elif param.name == 'optical_frame_id' and param.type_ == Parameter.Type.STRING:
                self.optical_frame_id = param.value
                # invalidate cached transforms that depended on the old optical frame
                self._optical_to_output_tf = None
                self._pc_to_optical_tf = None
                self._pc_to_optical_tf_src = None
            elif param.name == 'depth_box_thickness' and param.type_ == Parameter.Type.DOUBLE:
                self.depth_box_thickness = param.value
        return result

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
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()