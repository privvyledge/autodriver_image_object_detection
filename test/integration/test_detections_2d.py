"""End-to-end test of the 2D half of `create_detections_and_project`.

Drives the real node with `project_to_3d=False` and a fixed fake Ultralytics
result, so the output is deterministic — unlike a bag replay, where a stateful
tracker and an arrival-time synchronizer make two runs of identical code differ.

Small local fakes rather than mocks, matching the convention in `test/`.
"""
import numpy as np
import pytest
import torch


class FakeBoxes:
    def __init__(self, xywh, cls, conf, track_ids):
        self.xywh = torch.tensor(xywh, dtype=torch.float32).reshape(-1, 4)
        self.cls = torch.tensor(cls, dtype=torch.float32)
        self.conf = torch.tensor(conf, dtype=torch.float32)
        self.id = None if track_ids is None else torch.tensor(track_ids, dtype=torch.float32)
        self.is_track = track_ids is not None

    @property
    def shape(self):
        return self.xywh.shape

    def __len__(self):
        return self.xywh.shape[0]

    def cpu(self):
        return self


class FakeMasks:
    def __init__(self, data, xys):
        self.data = data
        self.xy = xys

    def cpu(self):
        return self


class FakeResult:
    def __init__(self, boxes, masks, names, orig_shape, canvas):
        self.boxes = boxes
        self.masks = masks
        self.names = names
        self.orig_shape = orig_shape
        self.keypoints = None
        self.obb = None
        self.probs = None
        self._canvas = canvas

    def plot(self, **kwargs):
        return self._canvas.copy()

    def cpu(self):
        return self


BOXES = [[30.0, 20.0, 10.0, 8.0], [70.0, 40.0, 16.0, 12.0], [50.0, 50.0, 6.0, 6.0]]
CLS = [0.0, 2.0, 56.0]
CONF = [0.91, 0.62, 0.44]
TRACK_IDS = [5, 9, 12]
NAMES = {0: 'person', 2: 'car', 56: 'chair'}


def build_result():
    rng = np.random.default_rng(1234)
    canvas = rng.integers(0, 255, size=(64, 96, 3)).astype(np.uint8)
    mask_data = torch.zeros((3, 64, 96), dtype=torch.float32)
    mask_data[0, 16:24, 25:35] = 1.0
    mask_data[1, 34:46, 62:78] = 1.0
    mask_data[2, 47:53, 47:53] = 1.0
    masks = FakeMasks(mask_data, [np.array([[25.0, 16.0], [35.0, 24.0]]),
                                  np.array([[62.0, 34.0], [78.0, 46.0]]),
                                  np.array([[47.0, 47.0], [53.0, 53.0]])])
    boxes = FakeBoxes(BOXES, CLS, CONF, TRACK_IDS)
    return FakeResult(boxes, masks, NAMES, (64, 96), canvas)


@pytest.fixture
def node_and_header(make_node, model_path):
    from autodriver_image_object_detection.yolo_detection_node import ImageObstacleDetectionNode
    from std_msgs.msg import Header

    node = make_node(
        ImageObstacleDetectionNode,
        model_path=model_path,
        use_gpu=False,
        half_precision=False,
        project_to_3d=False,
        use_depth=False,
        use_pointcloud=False,
        publish_debug_image=True,
        track_2d=True,
        plot_tracks=True,
    )
    header = Header()
    header.frame_id = 'camera_color_optical_frame'
    header.stamp.sec = 1700000000
    header.stamp.nanosec = 250000000
    node.frame_ids['rgb'] = header.frame_id
    node.msg_metadata['rgb'] = {'msg_timestamp': header.stamp}
    return node, header


class TestCreateDetectionsAndProject:
    def test_header_is_copied_from_the_image(self, node_and_header):
        node, header = node_and_header
        msg, _ = node.create_detections_and_project([build_result()], header)
        assert msg.header.frame_id == header.frame_id
        assert msg.header.stamp.sec == header.stamp.sec
        assert msg.header.stamp.nanosec == header.stamp.nanosec

    def test_every_box_becomes_a_detection(self, node_and_header):
        node, header = node_and_header
        msg, _ = node.create_detections_and_project([build_result()], header)
        assert len(msg.detections) == len(BOXES)

    def test_detection_fields_are_in_pixel_space(self, node_and_header):
        node, header = node_and_header
        msg, _ = node.create_detections_and_project([build_result()], header)
        for det, (cx, cy, w, h), cls, conf in zip(msg.detections, BOXES, CLS, CONF):
            hyp = det.results[0].hypothesis
            assert hyp.class_id == NAMES[int(cls)]
            assert hyp.score == pytest.approx(conf, abs=1e-6)
            assert det.bbox.center.position.x == pytest.approx(cx, abs=1e-6)
            assert det.bbox.center.position.y == pytest.approx(cy, abs=1e-6)
            assert det.bbox.size_x == pytest.approx(w, abs=1e-6)
            assert det.bbox.size_y == pytest.approx(h, abs=1e-6)

    def test_track_ids_are_carried_through(self, node_and_header):
        node, header = node_and_header
        msg, _ = node.create_detections_and_project([build_result()], header)
        assert [int(d.id) for d in msg.detections] == TRACK_IDS

    def test_mask_composite_covers_every_instance(self, node_and_header):
        node, header = node_and_header
        _, mask = node.create_detections_and_project([build_result()], header)
        assert mask is not None
        assert mask.shape == (64, 96)
        assert mask.dtype == np.uint8
        # One pixel inside each instance mask, and one outside all three.
        assert mask[20, 30] == 255
        assert mask[40, 70] == 255
        assert mask[50, 50] == 255
        assert mask[0, 0] == 0

    def test_track_history_accumulates_across_frames(self, node_and_header):
        node, header = node_and_header
        result = build_result()
        node.create_detections_and_project([result], header)
        after_one = {k: list(v) for k, v in node.track_history.items()}
        node.create_detections_and_project([result], header)

        assert sorted(after_one) == sorted(TRACK_IDS)
        for tid, (cx, cy, _, _) in zip(TRACK_IDS, BOXES):
            assert after_one[tid] == [pytest.approx((cx, cy))]
            assert node.track_history[tid] == [pytest.approx((cx, cy))] * 2

    def test_annotated_image_is_produced_when_a_consumer_exists(self, node_and_header):
        node, header = node_and_header
        node.create_detections_and_project([build_result()], header)
        assert node.detection_image is not None
        assert node.detection_image.shape == (64, 96, 3)

    def test_empty_result_yields_no_detections(self, node_and_header):
        node, header = node_and_header
        empty = FakeResult(FakeBoxes([], [], [], None), None, NAMES, (64, 96),
                           np.zeros((64, 96, 3), dtype=np.uint8))
        msg, mask = node.create_detections_and_project([empty], header)
        assert len(msg.detections) == 0
        assert mask is None
