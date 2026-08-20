import uuid
from collections import deque

import numpy as np
import torch
from vision_msgs.msg import Detection2D, Detection2DArray, Detection3D, Detection3DArray, ObjectHypothesisWithPose
from geometry_msgs.msg import Point, Vector3, Pose, Quaternion
from derived_object_msgs.msg import Object, ObjectArray
from shape_msgs.msg import SolidPrimitive

try:
    from nav2_dynamic_msgs.msg import Obstacle, ObstacleArray
except ImportError:
    print("nav2_dynamic_msgs not found. ")


def pack_2d_detection(x, y, size_x, size_y, class_id, conf, id, theta=0.0):
    """Pack one detection into a vision_msgs/Detection2D.

    Position and size are PIXELS in the frame named by the enclosing
    Detection2DArray header.

    theta is the rotation of the box about its centre, in radians, in the image
    frame (x right, y DOWN). It is 0.0 for an axis-aligned box, in which case
    size_x/size_y are the axis-aligned width/height. When theta is non-zero the
    box is rotated, so size_x is the extent along theta and size_y the extent
    perpendicular to it -- consumers that assume an axis-aligned box must check
    theta before using the sizes.
    """
    detection = Detection2D()
    detection.bbox.center.position.x = float(x)
    detection.bbox.center.position.y = float(y)
    detection.bbox.center.theta = float(theta)
    detection.bbox.size_x = float(size_x)
    detection.bbox.size_y = float(size_y)
    detection.id = str(id)
    hypothesis = ObjectHypothesisWithPose()
    hypothesis.hypothesis.class_id = class_id
    hypothesis.hypothesis.score = float(conf)
    detection.results.append(hypothesis)
    return detection


def pack_nav2_obstacle_msg(x, y, size_x, size_y, class_id, conf, id=None, z=0.0, z_size=1.0):
    """Pack a metric 3D object into a nav2_dynamic_msgs/Obstacle.

    Position and size are METRES in the frame named by the enclosing
    ObstacleArray header — never pixels. Callers without a depth or
    pointcloud projection have no metric object to publish and should emit
    vision_msgs/Detection2DArray instead.
    """
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
    obstacle_msg.position.z = float(z)
    obstacle_msg.size.x = float(size_x)
    obstacle_msg.size.y = float(size_y)
    obstacle_msg.size.z = float(z_size)
    return obstacle_msg

def pack_derived_object_msg(x, y, size_x, size_y, class_id, conf, id=None, z=0.0, z_size=1.0,
                            quat=None, tracked=True):
    """Pack a metric 3D object into a derived_object_msgs/Object.

    Position and size are METRES in the frame named by the enclosing ObjectArray
    header — never pixels. Callers without a depth or pointcloud projection have
    no metric object to publish and should emit vision_msgs/Detection2DArray
    instead.

    shape is always SolidPrimitive.BOX with dimensions [size_x, size_y, z_size].
    polygon is left empty; twist and accel are always zero (no velocity estimate
    is produced here). quat orients the box when the caller has an OBB, else the
    orientation is identity. tracked selects OBJECT_TRACKED vs OBJECT_DETECTED.
    """
    obj = Object()
    # Convert first 4 bytes of the obstacle's UUID into a uint32 id.
    if id in (None, -1):
        id = 10000000
    obj.id = id

    # Only objects carrying a tracker-assigned id are TRACKED; a per-frame
    # detection without one is DETECTED.
    obj.detection_level = Object.OBJECT_TRACKED if tracked else Object.OBJECT_DETECTED

    obj.pose.position = Point(x=float(x), y=float(y), z=float(z))
    if quat is None:
        obj.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    else:
        obj.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3]))

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
    sp.dimensions = [float(size_x), float(size_y), z_size]
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


def make_deleteall_marker_array():
    """Return a MarkerArray containing a single DELETEALL marker.

    Publishing this clears all previously published markers in RViz — used on
    zero-detection frames so stale boxes don't linger when objects leave view.
    """
    from visualization_msgs.msg import Marker, MarkerArray
    marker_arr = MarkerArray()
    marker = Marker()
    marker.action = Marker.DELETEALL
    marker_arr.markers.append(marker)
    return marker_arr


def update_tracker_param(param_name, new_value, old_value):
    # assert that both types match
    if type(new_value) != type(old_value):
        raise ValueError(
            f'The type {type(new_value)} does not match the type {type(old_value)} for parameter {param_name}.')

    if new_value == old_value:
        return new_value

    if isinstance(new_value, float):
        if new_value >= 0.0:  # negative values are sentinels meaning "use default"
            return new_value
    elif isinstance(new_value, int):
        if new_value >= 0:
            return new_value
    elif isinstance(new_value, str):
        if new_value:
            return new_value
    else:
        # for bools and strings
        return new_value
    return old_value


def parse_yolo_results(result) -> list:
    """Batch-extract detection data from a single Ultralytics result object.

    Transfers all tensors to CPU once before iterating to avoid per-detection
    device round-trips.

    Returns:
        List of dicts with keys: bbox_xywh (np.ndarray, shape 4),
        cls_id (int), cls_name (str), conf (float), track_id (int, -1 if none).
    """
    detections = []
    boxes = result.boxes.cpu()
    if boxes.shape[0] < 1:
        return detections

    all_xywh = boxes.xywh.numpy()        # (N, 4)
    all_cls = boxes.cls.numpy().astype(int)  # (N,)
    all_conf = boxes.conf.numpy()         # (N,)
    track_ids = None
    if boxes.is_track and boxes.id is not None:
        track_ids = boxes.id.int().numpy()

    for i in range(len(all_xywh)):
        detections.append({
            'bbox_xywh': all_xywh[i],
            'cls_id': int(all_cls[i]),
            'cls_name': result.names.get(int(all_cls[i]), str(int(all_cls[i]))),
            'conf': float(all_conf[i]),
            'track_id': int(track_ids[i]) if track_ids is not None else -1,
        })
    return detections


def get_centroid_for_class(bbox_xywh, class_name: str) -> tuple:
    """Return (cx, cy) adjusted by class semantics.

    For 'person', returns bottom-center (x, y+h/2) because feet indicate ground
    position. All other classes return the box center (x, y).
    """
    x, y, w, h = bbox_xywh
    if class_name == 'person':
        return float(x), float(y + h / 2)
    return float(x), float(y)


class TrackHistory:
    """Bounded per-track centroid history backed by collections.deque.

    Usage:
        history = TrackHistory(max_len=30)
        history.update(track_id, (cx, cy))
        trail = history.get(track_id)   # deque of (x, y) tuples
    """

    def __init__(self, max_len: int = 30):
        self._history: dict = {}
        self.max_len = max_len

    def update(self, track_id: int, centroid: tuple) -> None:
        if track_id not in self._history:
            self._history[track_id] = deque(maxlen=self.max_len)
        self._history[track_id].append(centroid)

    def get(self, track_id: int) -> deque:
        return self._history.get(track_id, deque(maxlen=self.max_len))

    def __contains__(self, track_id: int) -> bool:
        return track_id in self._history
