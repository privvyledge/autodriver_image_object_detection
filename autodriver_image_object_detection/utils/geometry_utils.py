import cv2
import numpy as np

from collections import namedtuple

#: Result of polygon_axis(). angle is radians in [-pi/2, pi/2] in the image
#: frame (x right, y DOWN) and is a 180-degree-ambiguous axis, not a direction.
#: elongation is sqrt(lambda_major / lambda_minor) -- 1.0 for a circular blob,
#: larger the more elongated. major/minor are the extents of the silhouette
#: projected onto the two principal axes, in pixels. ellipse_major/minor are
#: 4*sqrt(lambda) of the equivalent ellipse, whose ratio IS elongation exactly.
PolygonAxis = namedtuple(
    'PolygonAxis',
    'cx cy angle elongation major minor ellipse_major ellipse_minor')


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

def polygon_axis(polygon_pts):
    """Principal axis of a filled polygon, from exact second moments.

    Uses Green's theorem over the polygon outline, NOT a PCA over the vertex
    set: a segmentation contour samples its outline unevenly, so vertex PCA is
    biased toward whichever edge the polygoniser happened to sample densely.

    Coordinates are image-frame pixels (x right, y DOWN), so the returned angle
    is in that frame and is a 180-degree-ambiguous *axis*, not a direction —
    nothing here distinguishes front from back.

    Args:
        polygon_pts: (N, 2) array of polygon vertices in pixels.

    Returns:
        A PolygonAxis, or None for a degenerate polygon (< 3 vertices or zero
        area) which the caller must treat as "no orientation available".
    """
    pts = np.asarray(polygon_pts, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 3:
        return None

    x, y = pts[:, 0], pts[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a2 = cross.sum()          # twice the signed area
    if abs(a2) < 1e-9:
        return None

    # Normalise winding in place. This is a no-op for the moments below and is
    # kept only so downstream extent/centroid values are read off a consistently
    # wound polygon: every numerator here carries the same per-edge `cross` as
    # the 1/a2 denominator, so reversing the point order flips both and the
    # moments are exactly invariant (verified on synthetic rectangles and on
    # real mask contours, both orders). An implementation that takes abs() of
    # the area breaks that cancellation and *does* come out 90 degrees off on
    # clockwise input -- which is the common case: contours from Ultralytics
    # `masks.xy` are clockwise in image coordinates. ensure_ccw_polygon is not
    # used here: it quantises to int32 and mask contours are sub-pixel.
    if a2 < 0:
        pts = pts[::-1]
        x, y = pts[:, 0], pts[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cross = x * yn - xn * y
        a2 = -a2

    cx = ((x + xn) * cross).sum() / (3.0 * a2)
    cy = ((y + yn) * cross).sum() / (3.0 * a2)

    x, y = x - cx, y - cy
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    a2 = cross.sum()
    if abs(a2) < 1e-9:
        return None

    mu20 = ((x * x + x * xn + xn * xn) * cross).sum() / (12.0 * a2)
    mu02 = ((y * y + y * yn + yn * yn) * cross).sum() / (12.0 * a2)
    mu11 = ((2 * x * y + x * yn + xn * y + 2 * xn * yn) * cross).sum() / (24.0 * a2)

    angle = 0.5 * np.arctan2(2.0 * mu11, mu20 - mu02)
    common = mu20 + mu02
    diff = np.hypot(mu20 - mu02, 2.0 * mu11)
    lam_major = 0.5 * (common + diff)
    lam_minor = 0.5 * (common - diff)
    if lam_minor <= 1e-12:
        # A polygon with no measurable width: report it as maximally elongated
        # rather than dividing by ~0, so callers gate on the same scalar.
        elongation = float('inf')
    else:
        elongation = float(np.sqrt(lam_major / lam_minor))

    # Extents of the silhouette itself along the two principal axes.
    ca, sa = np.cos(angle), np.sin(angle)
    proj_major = x * ca + y * sa
    proj_minor = -x * sa + y * ca

    return PolygonAxis(
        cx=float(cx),
        cy=float(cy),
        angle=float(angle),
        elongation=elongation,
        major=float(proj_major.max() - proj_major.min()),
        minor=float(proj_minor.max() - proj_minor.min()),
        ellipse_major=float(4.0 * np.sqrt(max(lam_major, 0.0))),
        ellipse_minor=float(4.0 * np.sqrt(max(lam_minor, 0.0))),
    )
