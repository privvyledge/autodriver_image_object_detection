"""Tests for utils/geometry_utils.py — polygon winding and point-in-polygon."""
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodriver_image_object_detection.utils.geometry_utils import (  # noqa: E402
    batch_points_in_polygon,
    ensure_ccw_polygon,
    point_in_polygon,
    polygon_axis,
)


# A 100x100 axis-aligned square, listed clockwise in image coords (y down).
SQUARE_CW = [(0, 0), (0, 100), (100, 100), (100, 0)]
# The same square, counter-clockwise in image coords.
SQUARE_CCW = [(0, 0), (100, 0), (100, 100), (0, 100)]


def _signed_area(pts):
    """Shoelace signed area of an (N, 1, 2) or (N, 2) point array."""
    a = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    return 0.5 * float(
        np.sum(a[:, 0] * np.roll(a[:, 1], -1) - np.roll(a[:, 0], -1) * a[:, 1])
    )


class TestEnsureCcwPolygon(unittest.TestCase):

    def test_returns_expected_shape_and_dtype(self):
        pts = ensure_ccw_polygon(SQUARE_CCW)
        self.assertEqual(pts.shape, (4, 1, 2))
        self.assertEqual(pts.dtype, np.int32)

    def test_ccw_input_is_unchanged(self):
        pts = ensure_ccw_polygon(SQUARE_CCW)
        np.testing.assert_array_equal(
            pts.reshape(-1, 2), np.array(SQUARE_CCW, dtype=np.int32))

    def test_cw_input_is_reversed(self):
        pts = ensure_ccw_polygon(SQUARE_CW)
        np.testing.assert_array_equal(
            pts.reshape(-1, 2), np.array(SQUARE_CW[::-1], dtype=np.int32))

    def test_output_is_always_positively_signed(self):
        for polygon in (SQUARE_CW, SQUARE_CCW):
            with self.subTest(polygon=polygon):
                self.assertGreater(_signed_area(ensure_ccw_polygon(polygon)), 0.0)

    def test_accepts_numpy_and_nested_shapes(self):
        flat = ensure_ccw_polygon(np.array(SQUARE_CCW, dtype=np.float32))
        nested = ensure_ccw_polygon(
            np.array(SQUARE_CCW, dtype=np.float32).reshape(-1, 1, 2))
        np.testing.assert_array_equal(flat, nested)

    def test_non_convex_polygon_orientation(self):
        # L-shape; winding normalisation must not depend on convexity.
        l_shape = [(0, 0), (0, 100), (50, 100), (50, 50), (100, 50), (100, 0)]
        self.assertGreater(_signed_area(ensure_ccw_polygon(l_shape)), 0.0)


class TestPointInPolygon(unittest.TestCase):

    def test_interior_point(self):
        for polygon in (SQUARE_CW, SQUARE_CCW):
            with self.subTest(polygon=polygon):
                self.assertTrue(point_in_polygon((50, 50), polygon))

    def test_exterior_point(self):
        for polygon in (SQUARE_CW, SQUARE_CCW):
            with self.subTest(polygon=polygon):
                self.assertFalse(point_in_polygon((150, 50), polygon))

    def test_boundary_counts_as_inside(self):
        # pointPolygonTest returns 0.0 on an edge; the helper uses >= 0.
        self.assertTrue(point_in_polygon((0, 50), SQUARE_CCW))
        self.assertTrue(point_in_polygon((0, 0), SQUARE_CCW))

    def test_accepts_float_coordinates(self):
        self.assertTrue(point_in_polygon((50.5, 50.5), SQUARE_CCW))

    def test_non_convex_notch_is_outside(self):
        l_shape = [(0, 0), (0, 100), (50, 100), (50, 50), (100, 50), (100, 0)]
        self.assertTrue(point_in_polygon((25, 75), l_shape))
        self.assertFalse(point_in_polygon((75, 75), l_shape))


class TestBatchPointsInPolygon(unittest.TestCase):

    def test_mask_matches_scalar_helper(self):
        points = np.array([[50, 50], [150, 50], [10, 10], [-5, 20]], dtype=np.float32)
        mask = batch_points_in_polygon(points, SQUARE_CCW)
        self.assertEqual(mask.dtype, bool)
        self.assertEqual(mask.shape, (4,))
        expected = [point_in_polygon(tuple(p), SQUARE_CCW) for p in points]
        self.assertEqual(list(mask), expected)

    def test_winding_order_does_not_change_result(self):
        points = np.array([[50, 50], [150, 150], [99, 1]], dtype=np.float32)
        np.testing.assert_array_equal(
            batch_points_in_polygon(points, SQUARE_CW),
            batch_points_in_polygon(points, SQUARE_CCW))

    def test_empty_input(self):
        mask = batch_points_in_polygon(np.zeros((0, 2), dtype=np.float32), SQUARE_CCW)
        self.assertEqual(mask.shape, (0,))


class TestPolygonAxis(unittest.TestCase):
    """polygon_axis() must read the axis off the filled shape, not the vertices."""

    @staticmethod
    def _rect(w, h, angle_deg, cx=100.0, cy=50.0):
        pts = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                        [w / 2, h / 2], [-w / 2, h / 2]])
        a = math.radians(angle_deg)
        c, s = math.cos(a), math.sin(a)
        return pts @ np.array([[c, -s], [s, c]]).T + np.array([cx, cy])

    def test_rectangle_axis_and_extents_are_exact(self):
        for deg in (0.0, 30.0, -45.0, 80.0):
            with self.subTest(deg=deg):
                axis = polygon_axis(self._rect(40.0, 10.0, deg))
                self.assertAlmostEqual(math.degrees(axis.angle), deg, places=6)
                # A 4:1 rectangle has sqrt(lambda_major/lambda_minor) == 4.
                self.assertAlmostEqual(axis.elongation, 4.0, places=6)
                self.assertAlmostEqual(axis.major, 40.0, places=6)
                self.assertAlmostEqual(axis.minor, 10.0, places=6)
                self.assertAlmostEqual(axis.cx, 100.0, places=6)
                self.assertAlmostEqual(axis.cy, 50.0, places=6)

    def test_winding_order_does_not_change_the_axis(self):
        """Interop vector: clockwise input must give +30 deg, elongation 4.

        Real contours from Ultralytics `masks.xy` are clockwise in image
        coordinates, so this is the ordinary case, not an edge case. The
        absolute values are asserted and not just ccw == cw: an implementation
        that loses the sign of the signed area agrees with itself in both
        orders while sitting exactly 90 degrees away from this.
        """
        poly = self._rect(40.0, 10.0, 30.0)
        ccw, cw = polygon_axis(poly), polygon_axis(poly[::-1])
        self.assertAlmostEqual(ccw.angle, cw.angle, places=9)
        self.assertAlmostEqual(ccw.elongation, cw.elongation, places=9)
        self.assertAlmostEqual(math.degrees(cw.angle), 30.0, places=6)
        self.assertAlmostEqual(cw.elongation, 4.0, places=6)

    def test_ellipse_extents_ratio_equals_elongation(self):
        axis = polygon_axis(self._rect(40.0, 10.0, 17.0))
        self.assertAlmostEqual(
            axis.ellipse_major / axis.ellipse_minor, axis.elongation, places=9)

    def test_circular_blob_has_elongation_one(self):
        t = np.linspace(0.0, 2 * np.pi, 256, endpoint=False)
        axis = polygon_axis(np.c_[30 * np.cos(t), 30 * np.sin(t)])
        self.assertAlmostEqual(axis.elongation, 1.0, places=4)

    def test_uneven_vertex_sampling_does_not_bias_the_axis(self):
        """The reason for exact moments over vertex PCA.

        Densely resampling one long edge of a rectangle moves a vertex PCA but
        must leave the filled shape's moments untouched.
        """
        poly = self._rect(40.0, 10.0, 0.0)
        dense_edge = np.c_[np.linspace(80.0, 120.0, 50), np.full(50, 45.0)]
        resampled = np.vstack([dense_edge, poly[2:4]])
        self.assertAlmostEqual(
            polygon_axis(resampled).angle, polygon_axis(poly).angle, places=6)

    def test_degenerate_polygons_return_none(self):
        self.assertIsNone(polygon_axis(np.array([[0.0, 0.0], [1.0, 1.0]])))
        self.assertIsNone(polygon_axis(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])))


if __name__ == '__main__':
    unittest.main()
