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


def pack_nav2_obstacle_msg(x, y, size_x, size_y, class_id, conf, id=None, z_size=1.0):
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
    obstacle_msg.size.z = float(z_size)  # 0.0
    return obstacle_msg

def pack_derived_object_msg(x, y, size_x, size_y, class_id, conf, id=None, z_size=1.0):
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
