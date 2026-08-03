"""Tests for utils/common.py — message packers, tracker params, result parsing.

The packers need the ROS message packages on the path but no ROS graph, so these
run without a running node.
"""
import os
import sys
import unittest
import uuid

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from derived_object_msgs.msg import Object  # noqa: E402
from shape_msgs.msg import SolidPrimitive  # noqa: E402
from visualization_msgs.msg import Marker  # noqa: E402

from autodriver_image_object_detection.utils.common import (  # noqa: E402
    TrackHistory,
    get_centroid_for_class,
    make_deleteall_marker_array,
    pack_2d_detection,
    pack_derived_object_msg,
    parse_yolo_results,
    update_tracker_param,
)


class FakeBoxes:
    """Minimal stand-in for ultralytics.engine.results.Boxes.

    parse_yolo_results only touches .cpu(), .shape, .xywh, .cls, .conf,
    .is_track and .id, so the real class is not needed to test the parsing.
    """

    def __init__(self, xywh, cls, conf, track_ids=None):
        self.xywh = torch.tensor(xywh, dtype=torch.float32).reshape(-1, 4)
        self.cls = torch.tensor(cls, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.id = None if track_ids is None else torch.tensor(track_ids, dtype=torch.float32)
        self.is_track = track_ids is not None

    @property
    def shape(self):
        return self.xywh.shape

    def cpu(self):
        return self


class FakeResult:
    """Minimal stand-in for a single ultralytics Results object."""

    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


class TestPack2dDetection(unittest.TestCase):

    def test_fields_are_populated_and_cast_to_float(self):
        det = pack_2d_detection(10, 20, 30, 40, 'car', 0.9, 7)
        self.assertAlmostEqual(det.bbox.center.position.x, 10.0)
        self.assertAlmostEqual(det.bbox.center.position.y, 20.0)
        self.assertAlmostEqual(det.bbox.size_x, 30.0)
        self.assertAlmostEqual(det.bbox.size_y, 40.0)
        self.assertEqual(det.id, '7')
        self.assertEqual(len(det.results), 1)
        self.assertEqual(det.results[0].hypothesis.class_id, 'car')
        self.assertAlmostEqual(det.results[0].hypothesis.score, 0.9, places=6)

    def test_accepts_numpy_scalars(self):
        det = pack_2d_detection(
            np.float32(1.5), np.float32(2.5), np.float32(3.5), np.float32(4.5),
            'person', np.float32(0.5), -1)
        self.assertAlmostEqual(det.bbox.center.position.x, 1.5)
        self.assertEqual(det.id, '-1')


class TestPackDerivedObjectMsg(unittest.TestCase):

    def test_shape_is_a_box_with_xyz_dimensions(self):
        obj = pack_derived_object_msg(1.0, 2.0, 3.0, 4.0, 'car', 0.9, id=1,
                                      z=5.0, z_size=6.0)
        self.assertEqual(obj.shape.type, SolidPrimitive.BOX)
        self.assertEqual(list(obj.shape.dimensions), [3.0, 4.0, 6.0])
        self.assertAlmostEqual(obj.pose.position.z, 5.0)

    def test_polygon_twist_and_accel_are_empty_or_zero(self):
        obj = pack_derived_object_msg(1.0, 2.0, 3.0, 4.0, 'car', 0.9, id=1)
        self.assertEqual(len(obj.polygon.points), 0)
        for v in (obj.twist.linear, obj.twist.angular,
                  obj.accel.linear, obj.accel.angular):
            self.assertEqual((v.x, v.y, v.z), (0.0, 0.0, 0.0))

    def test_identity_orientation_when_no_quaternion(self):
        obj = pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=1)
        q = obj.pose.orientation
        self.assertEqual((q.x, q.y, q.z, q.w), (0.0, 0.0, 0.0, 1.0))

    def test_quaternion_is_applied_when_given(self):
        obj = pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=1,
                                      quat=(0.0, 0.0, 0.7071, 0.7071))
        self.assertAlmostEqual(obj.pose.orientation.z, 0.7071, places=4)
        self.assertAlmostEqual(obj.pose.orientation.w, 0.7071, places=4)

    def test_detection_level_follows_tracked_flag(self):
        tracked = pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.9,
                                          id=3, tracked=True)
        untracked = pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.9,
                                            id=3, tracked=False)
        self.assertEqual(tracked.detection_level, Object.OBJECT_TRACKED)
        self.assertEqual(untracked.detection_level, Object.OBJECT_DETECTED)

    def test_class_name_maps_to_classification(self):
        cases = {
            'car': Object.CLASSIFICATION_CAR,
            'truck': Object.CLASSIFICATION_TRUCK,
            'person': Object.CLASSIFICATION_PEDESTRIAN,
            'motorcycle': Object.CLASSIFICATION_MOTORCYCLE,
            'bicycle': Object.CLASSIFICATION_BIKE,
            'traffic light': Object.CLASSIFICATION_UNKNOWN,
        }
        for name, expected in cases.items():
            with self.subTest(name=name):
                obj = pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, name, 0.9, id=1)
                self.assertEqual(obj.classification, expected)

    def test_object_classified_follows_confidence_threshold(self):
        self.assertTrue(
            pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.26, id=1).object_classified)
        self.assertFalse(
            pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.25, id=1).object_classified)

    def test_certainty_scales_confidence_to_0_255(self):
        self.assertEqual(
            pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 1.0, id=1).classification_certainty,
            255)
        self.assertEqual(
            pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.0, id=1).classification_certainty,
            0)

    def test_missing_id_falls_back_to_sentinel(self):
        for missing in (None, -1):
            with self.subTest(id=missing):
                obj = pack_derived_object_msg(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=missing)
                self.assertEqual(obj.id, 10000000)


class TestPackNav2ObstacleMsg(unittest.TestCase):

    def setUp(self):
        try:
            from autodriver_image_object_detection.utils.common import pack_nav2_obstacle_msg
        except ImportError:  # pragma: no cover
            self.skipTest('nav2_dynamic_msgs not available')
        self.pack = pack_nav2_obstacle_msg
        try:
            self.pack(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=1)
        except NameError:
            self.skipTest('nav2_dynamic_msgs not available')

    def test_position_and_size_are_metres(self):
        obs = self.pack(1.0, 2.0, 3.0, 4.0, 'car', 0.9, id=1, z=5.0, z_size=6.0)
        self.assertEqual((obs.position.x, obs.position.y, obs.position.z), (1.0, 2.0, 5.0))
        self.assertEqual((obs.size.x, obs.size.y, obs.size.z), (3.0, 4.0, 6.0))
        self.assertAlmostEqual(obs.score, 0.9, places=6)

    def test_track_id_produces_a_deterministic_uuid(self):
        first = self.pack(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=42)
        second = self.pack(9.0, 9.0, 2.0, 2.0, 'car', 0.5, id=42)
        self.assertEqual(list(first.uuid.uuid), list(second.uuid.uuid))
        self.assertEqual(bytes(first.uuid.uuid), uuid.UUID(int=42).bytes)

    def test_missing_id_produces_a_random_uuid(self):
        for missing in (None, -1):
            with self.subTest(id=missing):
                a = self.pack(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=missing)
                b = self.pack(0.0, 0.0, 1.0, 1.0, 'car', 0.9, id=missing)
                self.assertEqual(len(a.uuid.uuid), 16)
                self.assertNotEqual(list(a.uuid.uuid), list(b.uuid.uuid))


class TestMakeDeleteallMarkerArray(unittest.TestCase):

    def test_contains_a_single_deleteall_marker(self):
        arr = make_deleteall_marker_array()
        self.assertEqual(len(arr.markers), 1)
        self.assertEqual(arr.markers[0].action, Marker.DELETEALL)


class TestUpdateTrackerParam(unittest.TestCase):

    def test_type_mismatch_raises(self):
        with self.assertRaises(ValueError):
            update_tracker_param('thresh', 1, 0.5)

    def test_unchanged_value_is_returned_as_is(self):
        self.assertEqual(update_tracker_param('thresh', 0.5, 0.5), 0.5)

    def test_valid_float_is_accepted(self):
        self.assertEqual(update_tracker_param('thresh', 0.8, 0.5), 0.8)

    def test_negative_float_sentinel_keeps_old_value(self):
        self.assertEqual(update_tracker_param('thresh', -1.0, 0.5), 0.5)

    def test_zero_is_a_valid_value_not_a_sentinel(self):
        self.assertEqual(update_tracker_param('thresh', 0.0, 0.5), 0.0)
        self.assertEqual(update_tracker_param('frames', 0, 30), 0)

    def test_negative_int_sentinel_keeps_old_value(self):
        self.assertEqual(update_tracker_param('frames', -5, 30), 30)

    def test_empty_string_keeps_old_value(self):
        self.assertEqual(update_tracker_param('name', '', 'bytetrack'), 'bytetrack')
        self.assertEqual(update_tracker_param('name', 'botsort', 'bytetrack'), 'botsort')

    def test_bool_toggle_is_accepted(self):
        self.assertIs(update_tracker_param('gmc', True, False), True)
        self.assertIs(update_tracker_param('gmc', False, True), False)


class TestParseYoloResults(unittest.TestCase):

    NAMES = {0: 'person', 2: 'car'}

    def test_empty_result_returns_empty_list(self):
        result = FakeResult(FakeBoxes([], [], []), self.NAMES)
        self.assertEqual(parse_yolo_results(result), [])

    def test_untracked_detections_get_sentinel_track_id(self):
        boxes = FakeBoxes([[10, 20, 30, 40], [50, 60, 70, 80]], [0, 2], [0.9, 0.5])
        dets = parse_yolo_results(FakeResult(boxes, self.NAMES))
        self.assertEqual(len(dets), 2)
        self.assertEqual([d['track_id'] for d in dets], [-1, -1])
        self.assertEqual([d['cls_name'] for d in dets], ['person', 'car'])
        self.assertEqual([d['cls_id'] for d in dets], [0, 2])
        np.testing.assert_allclose(dets[0]['bbox_xywh'], [10, 20, 30, 40])
        self.assertAlmostEqual(dets[1]['conf'], 0.5, places=6)

    def test_tracked_detections_carry_track_ids(self):
        boxes = FakeBoxes([[10, 20, 30, 40], [50, 60, 70, 80]], [0, 2], [0.9, 0.5],
                          track_ids=[3, 11])
        dets = parse_yolo_results(FakeResult(boxes, self.NAMES))
        self.assertEqual([d['track_id'] for d in dets], [3, 11])

    def test_is_track_true_but_id_none_falls_back_to_sentinel(self):
        # Ultralytics leaves .id None on the first tracked frame.
        boxes = FakeBoxes([[10, 20, 30, 40]], [0], [0.9], track_ids=[1])
        boxes.id = None
        dets = parse_yolo_results(FakeResult(boxes, self.NAMES))
        self.assertEqual(dets[0]['track_id'], -1)

    def test_unknown_class_id_falls_back_to_its_string_form(self):
        boxes = FakeBoxes([[10, 20, 30, 40]], [99], [0.9])
        dets = parse_yolo_results(FakeResult(boxes, self.NAMES))
        self.assertEqual(dets[0]['cls_name'], '99')


class TestGetCentroidForClass(unittest.TestCase):

    def test_person_uses_bottom_center(self):
        self.assertEqual(get_centroid_for_class((10, 20, 30, 40), 'person'), (10.0, 40.0))

    def test_other_classes_use_box_center(self):
        for name in ('car', 'truck', 'bicycle', ''):
            with self.subTest(name=name):
                self.assertEqual(get_centroid_for_class((10, 20, 30, 40), name), (10.0, 20.0))

    def test_returns_python_floats_from_numpy_input(self):
        cx, cy = get_centroid_for_class(np.array([10, 20, 30, 40], dtype=np.float32), 'person')
        self.assertIsInstance(cx, float)
        self.assertIsInstance(cy, float)


class TestTrackHistory(unittest.TestCase):

    def test_update_then_get(self):
        history = TrackHistory(max_len=5)
        history.update(1, (10.0, 20.0))
        history.update(1, (11.0, 21.0))
        self.assertEqual(list(history.get(1)), [(10.0, 20.0), (11.0, 21.0)])

    def test_history_is_bounded_by_max_len(self):
        history = TrackHistory(max_len=3)
        for i in range(10):
            history.update(1, (float(i), 0.0))
        trail = list(history.get(1))
        self.assertEqual(len(trail), 3)
        self.assertEqual(trail, [(7.0, 0.0), (8.0, 0.0), (9.0, 0.0)])

    def test_tracks_are_independent(self):
        history = TrackHistory(max_len=5)
        history.update(1, (1.0, 1.0))
        history.update(2, (2.0, 2.0))
        self.assertEqual(list(history.get(1)), [(1.0, 1.0)])
        self.assertEqual(list(history.get(2)), [(2.0, 2.0)])

    def test_get_unknown_track_returns_empty_without_creating_it(self):
        history = TrackHistory(max_len=5)
        self.assertEqual(list(history.get(99)), [])
        self.assertNotIn(99, history)

    def test_contains(self):
        history = TrackHistory(max_len=5)
        self.assertNotIn(1, history)
        history.update(1, (0.0, 0.0))
        self.assertIn(1, history)


if __name__ == '__main__':
    unittest.main()
