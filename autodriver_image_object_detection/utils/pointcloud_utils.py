import numpy as np
import torch
import torch.utils.dlpack

try:
    import open3d as o3d
    import open3d.core as o3c
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

try:
    from sensor_msgs_py import point_cloud2 as pc2_util
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


def convert_to_open3d_tensor(input_array, device='CPU:0'):
    if isinstance(input_array, np.ndarray):
        input_array = o3c.Tensor(input_array, device=device)
    if isinstance(input_array, torch.Tensor):
        input_array = o3c.Tensor.from_dlpack(
            torch.utils.dlpack.to_dlpack(input_array)
        ).to(device=device)
    return input_array


def unpack_pointcloud_message(ros_cloud, fields=('x', 'y', 'z'), skip_nans: bool = True):
    """Convert a sensor_msgs/PointCloud2 message to a numpy structured array.

    Args:
        ros_cloud: sensor_msgs/PointCloud2 message.
        fields: field names to extract. Default ('x','y','z').
        skip_nans: drop points where any requested field is NaN.

    Returns:
        (frame_id, timestamp, points_np) where points_np is float32 (N, len(fields)),
        or (None, None, None) on error.
    """
    if not ROS_AVAILABLE:
        raise ImportError("sensor_msgs_py is required for unpack_pointcloud_message")

    try:
        frame_id = ros_cloud.header.frame_id
        timestamp = ros_cloud.header.stamp
        available = {f.name for f in ros_cloud.fields}
        use_fields = tuple(f for f in fields if f in available)
        if not use_fields:
            return None, None, None

        cloud_array = pc2_util.read_points_numpy(
            ros_cloud, field_names=use_fields, skip_nans=skip_nans
        )
        if cloud_array.ndim == 1:
            cloud_array = cloud_array.reshape(-1, 1)
        points_np = cloud_array.astype(np.float32)
        return frame_id, timestamp, points_np
    except Exception:
        return None, None, None


def preprocess_pointcloud(
    points: np.ndarray,
    max_distance: float = None,
    z_min: float = None,
    z_max: float = None,
) -> np.ndarray:
    """Filter a pointcloud array in-place by distance and Z bounds.

    Args:
        points: (N, ≥3) float32 array with columns [x, y, z, ...].
        max_distance: discard points whose XY distance from origin exceeds this.
        z_min: discard points below this Z height (metres).
        z_max: discard points above this Z height (metres).

    Returns:
        Filtered (M, K) float32 array.
    """
    if points is None or len(points) == 0:
        return points

    mask = np.ones(len(points), dtype=bool)
    if max_distance is not None:
        xy_dist = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
        mask &= xy_dist <= max_distance
    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    return points[mask]


def project_depth_to_3d(bbox_xywh, depth_image: np.ndarray, camera_model, depth_scale: float, mask=None):
    """Back-project a single 2D detection to 3D using a depth image.

    This is the camera-intrinsics-only path (no TF lookup). The caller is
    responsible for any depth-to-RGB frame alignment before passing depth_image.

    Steps:
        1. Crop depth to the segmentation mask or bounding-box ROI.
        2. Scale to metres.
        3. Find robust depth estimate (median of valid pixels).
        4. Unproject via the camera model's ray.

    Args:
        bbox_xywh: (x, y, w, h) bounding-box centre + size in pixels.
        depth_image: HxW numpy array in native sensor units.
        camera_model: image_geometry.PinholeCameraModel.
        depth_scale: divide raw pixel values by this to get metres
            (1000.0 for 16UC1 mm, 1.0 for 32FC1 m).
        mask: optional Ultralytics mask object; if provided, depth is sampled
            only within the segmentation mask instead of the full bbox.

    Returns:
        (x3d, y3d, z3d) in the camera frame, or None on failure.
    """
    cx, cy = int(bbox_xywh[0]), int(bbox_xywh[1])
    w, h = int(bbox_xywh[2]), int(bbox_xywh[3])

    if mask is not None:
        mask_data = mask.data.cpu().numpy().astype(np.uint8)[0] * 255
        roi = cv2_bitwise_and_depth(depth_image, mask_data)
    else:
        u0 = max(cx - w // 2, 0)
        u1 = min(cx + w // 2, depth_image.shape[1] - 1)
        v0 = max(cy - h // 2, 0)
        v1 = min(cy + h // 2, depth_image.shape[0] - 1)
        roi = depth_image[v0:v1, u0:u1]

    roi_m = roi.astype(np.float32) / depth_scale
    valid = roi_m[roi_m > 0]
    if valid.size == 0:
        return None

    z = float(np.median(valid))
    ray = camera_model.projectPixelTo3dRay((cx, cy))
    scale = z / ray[2]
    return float(ray[0] * scale), float(ray[1] * scale), z


def cv2_bitwise_and_depth(depth_image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a uint8 mask to a depth image of arbitrary dtype."""
    import cv2
    if depth_image.dtype == np.uint16:
        return cv2.bitwise_and(depth_image, depth_image, mask=mask)
    # float32 — cv2.bitwise_and doesn't support float; use numpy masking instead
    out = depth_image.copy()
    out[mask == 0] = 0
    return out


def cluster_pointcloud(points_xyz: np.ndarray, eps: float = 0.3, min_samples: int = 5):
    """Cluster a pointcloud using DBSCAN.

    Requires scikit-learn.

    Args:
        points_xyz: (N, 3) float32 XYZ array.
        eps: DBSCAN neighbourhood radius (metres).
        min_samples: minimum points to form a core sample.

    Returns:
        List of (M, 3) numpy arrays, one per cluster (noise excluded).
    """
    try:
        from sklearn.cluster import DBSCAN
    except ImportError:
        raise ImportError("scikit-learn is required for cluster_pointcloud")

    if points_xyz is None or len(points_xyz) < min_samples:
        return []

    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points_xyz)
    clusters = []
    for label in set(labels):
        if label == -1:  # noise
            continue
        clusters.append(points_xyz[labels == label])
    return clusters