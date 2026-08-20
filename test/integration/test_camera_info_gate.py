"""`single_stream_detector` must detect without ever receiving a CameraInfo.

This node publishes pixel-space `Detection2DArray`; its inference size comes from
the frame itself or from the `image_dimensions` parameter, so it needs no camera
intrinsics. It used to drop every frame until a `CameraInfo` arrived, which meant
any source without a `CameraInfo` publisher — a video file, or a bag recorded
without the topic — produced *no detections at all* behind a single one-shot
warning.

Nothing pinned that behaviour in either direction, so the regression these tests
guard is a re-added precondition on `self.camera_info`.
"""
import queue
import threading

import numpy as np
import pytest


def _make(make_node, model_path):
    from autodriver_image_object_detection.single_stream_detector import SingleStreamDetector

    return make_node(
        SingleStreamDetector,
        model_path=model_path,
        use_gpu=False,
        half_precision=False,
        publish_debug_image=False,
        show_image=False,
        track_2d=False,
        plot_tracks=False,
        use_image_dimensions=False,
        image_dimensions=[96, 160],
    )


def _feed_one_frame(node, timeout=30.0):
    """Push one image through the worker thread; return the published message."""
    published = queue.Queue()
    real_publish = node.detection_results_pub.publish

    def capture(msg):
        real_publish(msg)
        published.put(msg)

    node.detection_results_pub.publish = capture

    frame = np.zeros((96, 160, 3), dtype=np.uint8)
    # A mid-grey block so the frame is not degenerate; content is irrelevant here,
    # only that the worker reaches the publish call at all.
    frame[24:72, 40:120] = 128
    msg = node.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
    msg.header.frame_id = 'camera_color_optical_frame'

    node._image_queue.put(msg)
    try:
        return published.get(timeout=timeout)
    except queue.Empty:
        return None


class TestCameraInfoIsNotAPrecondition:
    def test_publishes_detections_with_no_camera_info(self, make_node, model_path):
        node = _make(make_node, model_path)
        assert node.camera_info is None, 'no CameraInfo was published to this node'

        msg = _feed_one_frame(node)

        assert msg is not None, (
            'no Detection2DArray was published — the worker is gating on CameraInfo again'
        )
        assert node.camera_info is None, 'the node must not require intrinsics to run'

    def test_camera_info_is_still_stored_when_it_does_arrive(self, make_node, model_path):
        """The subscription is kept for consumers that want intrinsics (ROI, undistort).

        Removing the *gate* must not remove the *storage*.
        """
        from sensor_msgs.msg import CameraInfo

        node = _make(make_node, model_path)
        info = CameraInfo()
        info.width, info.height = 160, 96
        node.camera_info_callback(info)

        assert node.camera_info is info

    def test_worker_thread_survives_the_frame(self, make_node, model_path):
        """A frame processed with camera_info None must not kill the worker."""
        node = _make(make_node, model_path)
        _feed_one_frame(node)

        alive = [t for t in threading.enumerate() if t is node._inference_thread]
        assert alive and alive[0].is_alive(), 'inference worker died processing the frame'


@pytest.mark.parametrize('attr', ['camera_info_sub', 'camera_info_callback'])
def test_camera_info_plumbing_is_retained(make_node, model_path, attr):
    node = _make(make_node, model_path)
    assert hasattr(node, attr)
