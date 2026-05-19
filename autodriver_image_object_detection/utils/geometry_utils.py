import cv2
import numpy as np


def ensure_ccw_polygon(pts) -> np.ndarray:
    """Return polygon points normalised to CCW winding order.

    cv2.pointPolygonTest is correct for any simple polygon, but CCW is the
    conventional positive orientation in image coordinates (y↓). User-drawn
    polygons may arrive in CW order; this normalises them.

    Uses the shoelace signed-area formula. In image coords (y increasing
    downward) a positive signed area corresponds to CCW winding.

    Returns:
        np.ndarray of shape (N, 1, 2) with dtype int32, suitable for
        cv2.pointPolygonTest and cv2.fillPoly.
    """
    pts_arr = np.array(pts, dtype=np.float32).reshape(-1, 2)
    signed_area = 0.5 * float(
        np.sum(
            pts_arr[:, 0] * np.roll(pts_arr[:, 1], -1)
            - np.roll(pts_arr[:, 0], -1) * pts_arr[:, 1]
        )
    )
    if signed_area < 0:
        pts_arr = pts_arr[::-1]
    return pts_arr.reshape(-1, 1, 2).astype(np.int32)


def point_in_polygon(point: tuple, polygon_pts) -> bool:
    """Return True if *point* is inside *polygon_pts*.

    Polygon winding order is normalised to CCW before testing.

    Args:
        point: (x, y) in image coordinates.
        polygon_pts: sequence of (x, y) pairs or any shape accepted by
            ensure_ccw_polygon.
    """
    pts = ensure_ccw_polygon(polygon_pts)
    return cv2.pointPolygonTest(pts, (float(point[0]), float(point[1])), False) >= 0


def batch_points_in_polygon(points: np.ndarray, polygon_pts) -> np.ndarray:
    """Return a boolean mask (length N) for which of N points fall inside polygon.

    Args:
        points: (N, 2) array of (x, y) coordinates.
        polygon_pts: polygon vertices accepted by ensure_ccw_polygon.

    Returns:
        np.ndarray of shape (N,) dtype bool.
    """
    pts = ensure_ccw_polygon(polygon_pts)
    return np.array(
        [cv2.pointPolygonTest(pts, (float(p[0]), float(p[1])), False) >= 0 for p in points],
        dtype=bool,
    )