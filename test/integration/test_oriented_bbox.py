"""Node-level test that the mask principal axis reaches Detection2D.bbox.

`publish_oriented_bbox` writes the instance mask's principal axis into
`bbox.center.theta` and replaces size_x/size_y with the extents of that rotated
box. This pins both the default (off, axis-aligned, theta == 0) and the two
size modes, using a fake Ultralytics result whose mask polygons are exact
rectangles so the expected angle and elongation are known in closed form.

Runs through `single_stream_detector`, which uses the base
`create_detections_array` unmodified.
"""
import math

import numpy as np
import pytest
import torch

from test_detections_2d import FakeBoxes, FakeMasks, FakeResult


NAMES = {2: 'car'}
#: (centre_x, centre_y, length, width, angle_deg) of each synthetic mask.
SHAPES = [(48.0, 32.0, 40.0, 10.0, 30.0), (120.0, 60.0, 24.0, 12.0, -20.0)]


def _rect_polygon(cx, cy, length, width, angle_deg):
    pts = np.array([[-length / 2, -width / 2], [length / 2, -width / 2],
                    [length / 2, width / 2], [-length / 2, width / 2]])
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return pts @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])


def build_result():
    canvas = np.zeros((96, 160, 3), dtype=np.uint8)
    polys = [_rect_polygon(*shape) for shape in SHAPES]
    # Axis-aligned boxes, deliberately unrelated to the mask extents so a test
    # cannot pass by accident when the oriented path is skipped.
    xywh = [[cx, cy, 50.0, 50.0] for cx, cy, _, _, _ in SHAPES]
    boxes = FakeBoxes(xywh, [2.0] * len(SHAPES), [0.9] * len(SHAPES), [1, 2])
    masks = FakeMasks(torch.zeros((len(SHAPES), 96, 160), dtype=torch.float32), polys)
    return FakeResult(boxes, masks, NAMES, (96, 160), canvas)


def _make(make_node, model_path, **overrides):
    from autodriver_image_object_detection.single_stream_detector import SingleStreamDetector
    from std_msgs.msg import Header

    node = make_node(
        SingleStreamDetector,
        model_path=model_path,
        use_gpu=False,
        half_precision=False,
        publish_debug_image=False,
        show_image=False,
        track_2d=True,
        plot_tracks=False,
        **overrides,
    )
    header = Header()
    header.frame_id = 'camera_color_optical_frame'
    return node, header


class TestOrientedBoundingBox:
    def test_off_by_default_keeps_axis_aligned_boxes(self, make_node, model_path):
        node, header = _make(make_node, model_path)
        assert node.publish_oriented_bbox is False
        msg, _, _ = node.create_detections_array(build_result(), header)
        for det in msg.detections:
            assert det.bbox.center.theta == 0.0
            assert det.bbox.size_x == pytest.approx(50.0)
            assert det.bbox.size_y == pytest.approx(50.0)

    def test_obb_mode_publishes_axis_and_silhouette_extents(self, make_node, model_path):
        node, header = _make(make_node, model_path, publish_oriented_bbox=True)
        msg, _, _ = node.create_detections_array(build_result(), header)
        assert len(msg.detections) == len(SHAPES)
        for det, (cx, cy, length, width, deg) in zip(msg.detections, SHAPES):
            # The centre stays the *bbox* centre; only theta and the sizes change.
            assert det.bbox.center.position.x == pytest.approx(cx, abs=1e-4)
            assert det.bbox.center.position.y == pytest.approx(cy, abs=1e-4)
            assert math.degrees(det.bbox.center.theta) == pytest.approx(deg, abs=1e-4)
            assert det.bbox.size_x == pytest.approx(length, abs=1e-4)
            assert det.bbox.size_y == pytest.approx(width, abs=1e-4)

    def test_moments_mode_size_ratio_is_the_elongation(self, make_node, model_path):
        node, header = _make(make_node, model_path, publish_oriented_bbox=True,
                             oriented_bbox_size_mode='moments')
        msg, _, _ = node.create_detections_array(build_result(), header)
        for det, (_, _, length, width, deg) in zip(msg.detections, SHAPES):
            assert math.degrees(det.bbox.center.theta) == pytest.approx(deg, abs=1e-4)
            # sqrt(lambda_major/lambda_minor) of a rectangle is its aspect ratio.
            assert det.bbox.size_x / det.bbox.size_y == pytest.approx(length / width,
                                                                     abs=1e-6)

    def test_detection_without_a_usable_mask_stays_axis_aligned(self, make_node, model_path):
        node, header = _make(make_node, model_path, publish_oriented_bbox=True)
        result = build_result()
        # A two-point outline has no area — polygon_axis must decline it rather
        # than emit a meaningless angle.
        result.masks.xy[1] = np.array([[10.0, 10.0], [20.0, 20.0]])
        msg, _, _ = node.create_detections_array(result, header)
        assert math.degrees(msg.detections[0].bbox.center.theta) == pytest.approx(30.0,
                                                                                 abs=1e-4)
        assert msg.detections[1].bbox.center.theta == 0.0
        assert msg.detections[1].bbox.size_x == pytest.approx(50.0)

    def test_no_masks_at_all_is_not_an_error(self, make_node, model_path):
        node, header = _make(make_node, model_path, publish_oriented_bbox=True)
        result = build_result()
        result.masks = None
        msg, _, _ = node.create_detections_array(result, header)
        assert [d.bbox.center.theta for d in msg.detections] == [0.0] * len(SHAPES)

    def test_size_mode_is_live_reconfigurable_and_validated(self, make_node, model_path):
        from rclpy.parameter import Parameter

        node, _ = _make(make_node, model_path)
        ok = node.parameter_change_callback(
            [Parameter('oriented_bbox_size_mode', Parameter.Type.STRING, 'moments')])
        assert ok.successful and node.oriented_bbox_size_mode == 'moments'

        bad = node.parameter_change_callback(
            [Parameter('oriented_bbox_size_mode', Parameter.Type.STRING, 'nonsense')])
        assert not bad.successful
        assert node.oriented_bbox_size_mode == 'moments'
