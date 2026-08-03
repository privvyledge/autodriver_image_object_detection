"""Tests for utils/imaging_utils.py — ROS Image/CompressedImage to OpenCV BGR."""
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from cv_bridge import CvBridge  # noqa: E402

from autodriver_image_object_detection.utils.imaging_utils import (  # noqa: E402
    parse_image_message,
)


class CollectingLogger:
    def __init__(self):
        self.errors = []

    def error(self, msg):
        self.errors.append(msg)


class ImagingTestBase(unittest.TestCase):

    def setUp(self):
        self.bridge = CvBridge()
        # A small image with distinct per-channel values so channel order is testable.
        self.bgr = np.zeros((4, 6, 3), dtype=np.uint8)
        self.bgr[:, :, 0] = 10   # B
        self.bgr[:, :, 1] = 20   # G
        self.bgr[:, :, 2] = 30   # R

    def _msg(self, image, encoding, frame_id='camera_optical_frame'):
        msg = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        msg.header.frame_id = frame_id
        msg.header.stamp.sec = 7
        msg.header.stamp.nanosec = 42
        return msg


class TestColourEncodings(ImagingTestBase):

    def test_bgr8_passes_through_unchanged(self):
        out = parse_image_message(self._msg(self.bgr, 'bgr8'), self.bridge)
        cv_image, encoding, _, _, msg_fmt, conversion, inverse, is_color, is_depth, codec = out
        np.testing.assert_array_equal(cv_image, self.bgr)
        self.assertEqual(encoding, 'bgr8')
        self.assertEqual(msg_fmt, 'bgr8')
        self.assertIsNone(conversion)
        self.assertIsNone(inverse)
        self.assertTrue(is_color)
        self.assertFalse(is_depth)
        self.assertIsNone(codec)

    def test_rgb8_is_converted_to_bgr(self):
        rgb = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGB)
        cv_image, _, _, _, msg_fmt, conversion, inverse, is_color, _, _ = parse_image_message(
            self._msg(rgb, 'rgb8'), self.bridge)
        np.testing.assert_array_equal(cv_image, self.bgr)
        self.assertEqual(msg_fmt, 'rgb8')
        self.assertEqual(conversion, cv2.COLOR_RGB2BGR)
        self.assertEqual(inverse, cv2.COLOR_BGR2RGB)
        self.assertTrue(is_color)

    def test_bgra8_drops_the_alpha_channel(self):
        bgra = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2BGRA)
        cv_image, _, _, _, msg_fmt, conversion, _, _, _, _ = parse_image_message(
            self._msg(bgra, 'bgra8'), self.bridge)
        self.assertEqual(cv_image.shape, self.bgr.shape)
        np.testing.assert_array_equal(cv_image, self.bgr)
        self.assertEqual(msg_fmt, 'bgra8')
        self.assertEqual(conversion, cv2.COLOR_BGRA2BGR)

    def test_rgba8_drops_alpha_and_reorders_channels(self):
        rgba = cv2.cvtColor(self.bgr, cv2.COLOR_BGR2RGBA)
        cv_image, _, _, _, msg_fmt, conversion, _, _, _, _ = parse_image_message(
            self._msg(rgba, 'rgba8'), self.bridge)
        np.testing.assert_array_equal(cv_image, self.bgr)
        self.assertEqual(msg_fmt, 'rgba8')
        self.assertEqual(conversion, cv2.COLOR_RGBA2BGR)

    def test_mono8_is_expanded_to_three_channels(self):
        gray = np.full((4, 6), 128, dtype=np.uint8)
        cv_image, _, _, _, msg_fmt, conversion, inverse, is_color, is_depth, _ = (
            parse_image_message(self._msg(gray, 'mono8'), self.bridge))
        self.assertEqual(cv_image.shape, (4, 6, 3))
        self.assertTrue((cv_image == 128).all())
        self.assertEqual(msg_fmt, 'mono8')
        self.assertEqual(conversion, cv2.COLOR_GRAY2BGR)
        self.assertEqual(inverse, cv2.COLOR_BGR2GRAY)
        self.assertFalse(is_color)
        self.assertFalse(is_depth)

    def test_8uc1_is_expanded_to_three_channels_via_passthrough(self):
        # cv_bridge rejects an 8UC1 -> mono8 conversion, so this path must ask
        # for passthrough and rely on COLOR_GRAY2BGR for the expansion.
        gray = np.full((4, 6), 77, dtype=np.uint8)
        cv_image, _, _, _, msg_fmt, conversion, inverse, is_color, is_depth, _ = (
            parse_image_message(self._msg(gray, '8UC1'), self.bridge))
        self.assertEqual(cv_image.shape, (4, 6, 3))
        self.assertTrue((cv_image == 77).all())
        self.assertEqual(msg_fmt, 'passthrough')
        self.assertEqual(conversion, cv2.COLOR_GRAY2BGR)
        self.assertEqual(inverse, cv2.COLOR_BGR2GRAY)
        self.assertFalse(is_color)
        self.assertFalse(is_depth)


class TestDepthEncodings(ImagingTestBase):

    def test_16uc1_is_flagged_as_depth_and_left_single_channel(self):
        depth = np.full((4, 6), 1500, dtype=np.uint16)
        cv_image, _, _, _, msg_fmt, conversion, _, is_color, is_depth, _ = (
            parse_image_message(self._msg(depth, '16UC1'), self.bridge,
                                depth_scale=1000.0))
        self.assertEqual(cv_image.shape, (4, 6))
        np.testing.assert_array_equal(cv_image, depth)
        self.assertEqual(msg_fmt, '16UC1')
        self.assertIsNone(conversion)
        self.assertFalse(is_color)
        self.assertTrue(is_depth)

    def test_32fc1_is_flagged_as_depth(self):
        depth = np.full((4, 6), 1.5, dtype=np.float32)
        cv_image, _, _, _, msg_fmt, _, _, _, is_depth, _ = parse_image_message(
            self._msg(depth, '32FC1'), self.bridge, depth_scale=1.0)
        np.testing.assert_allclose(cv_image, depth)
        self.assertEqual(msg_fmt, '32FC1')
        self.assertTrue(is_depth)

    def test_wrong_depth_scale_for_16uc1_raises(self):
        depth = np.full((4, 6), 1500, dtype=np.uint16)
        with self.assertRaises(AssertionError):
            parse_image_message(self._msg(depth, '16UC1'), self.bridge, depth_scale=1.0)

    def test_wrong_depth_scale_for_32fc1_raises(self):
        depth = np.full((4, 6), 1.5, dtype=np.float32)
        with self.assertRaises(AssertionError):
            parse_image_message(self._msg(depth, '32FC1'), self.bridge, depth_scale=1000.0)

    def test_none_depth_scale_skips_the_assertion(self):
        depth = np.full((4, 6), 1500, dtype=np.uint16)
        _, _, _, _, msg_fmt, _, _, _, is_depth, _ = parse_image_message(
            self._msg(depth, '16UC1'), self.bridge, depth_scale=None)
        self.assertEqual(msg_fmt, '16UC1')
        self.assertTrue(is_depth)


class TestHeaderPassthrough(ImagingTestBase):

    def test_frame_id_and_timestamp_are_returned(self):
        msg = self._msg(self.bgr, 'bgr8', frame_id='left_camera_optical')
        _, _, frame_id, stamp, _, _, _, _, _, _ = parse_image_message(msg, self.bridge)
        self.assertEqual(frame_id, 'left_camera_optical')
        self.assertEqual(stamp.sec, 7)
        self.assertEqual(stamp.nanosec, 42)


class TestCompressedImages(ImagingTestBase):

    def _compressed(self, image, fmt='bgr8; jpeg compressed bgr8'):
        msg = self.bridge.cv2_to_compressed_imgmsg(image, dst_format='jpg')
        msg.format = fmt
        msg.header.frame_id = 'camera_optical_frame'
        return msg

    def test_codec_and_encoding_are_parsed_from_the_format_string(self):
        msg = self._compressed(self.bgr, 'rgb8; jpeg compressed bgr8')
        cv_image, encoding, _, _, msg_fmt, _, _, _, _, codec = parse_image_message(
            msg, self.bridge, image_message_format='compressed')
        self.assertEqual(codec, 'jpeg')
        self.assertEqual(encoding, 'bgr8')
        self.assertEqual(msg_fmt, 'bgr8')
        self.assertEqual(cv_image.shape, self.bgr.shape)

    def test_png_codec_is_parsed(self):
        msg = self.bridge.cv2_to_compressed_imgmsg(self.bgr, dst_format='png')
        msg.format = 'bgr8; png compressed bgr8'
        _, _, _, _, _, _, _, _, _, codec = parse_image_message(
            msg, self.bridge, image_message_format='compressed')
        self.assertEqual(codec, 'png')


class TestUnsupportedEncoding(ImagingTestBase):

    def test_unknown_encoding_falls_back_to_passthrough_and_logs(self):
        image = np.zeros((4, 6, 3), dtype=np.uint16)
        msg = self._msg(image, '16UC3')
        logger = CollectingLogger()
        cv_image, _, _, _, msg_fmt, _, _, _, _, _ = parse_image_message(
            msg, self.bridge, logger=logger)
        self.assertEqual(msg_fmt, 'passthrough')
        self.assertEqual(cv_image.shape, image.shape)
        self.assertEqual(len(logger.errors), 1)
        self.assertIn('16UC3', logger.errors[0])


if __name__ == '__main__':
    unittest.main()
