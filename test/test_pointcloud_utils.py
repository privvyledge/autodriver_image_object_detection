"""Tests for the depth-surface selection used by both 3D projection paths.

The cases below are built to the two scales the nodes actually run at: a
RealSense D435i looking at indoor furniture a couple of metres away, and a
simulated driving scene with vehicles at 15-30 m. The same defaults must give a
sane view-axis extent in both, which is exactly what a fixed-size window could
not do.
"""
import numpy as np
import pytest

from autodriver_image_object_detection.utils.pointcloud_utils import select_object_depths


def surface(centre, thickness, n=400, seed=0):
    """Depth samples spread uniformly over an object's visible surface."""
    rng = np.random.default_rng(seed)
    return rng.uniform(centre - thickness / 2.0, centre + thickness / 2.0, n)


class TestEmptyAndDegenerate:
    def test_empty_returns_none(self):
        assert select_object_depths(np.array([])) is None

    def test_single_sample_has_zero_extent(self):
        z, lo, hi = select_object_depths(np.array([2.5]))
        assert z == pytest.approx(2.5)
        assert hi - lo == pytest.approx(0.0)

    def test_z_hi_never_below_z_lo(self):
        _, lo, hi = select_object_depths(np.array([1.0, 1.0, 1.0, 1.0]))
        assert hi >= lo

    def test_accepts_a_python_list(self):
        assert select_object_depths([1.0, 1.05, 1.1]) is not None

    def test_flattens_2d_input(self):
        z, _, _ = select_object_depths(np.full((10, 10), 3.0))
        assert z == pytest.approx(3.0)


class TestIndoorScale:
    """D435i: a chair ~0.35 m deep at 2 m, with a wall 1.5 m behind it."""

    def test_chair_alone_measures_its_own_depth(self):
        _, lo, hi = select_object_depths(surface(2.0, 0.35))
        assert hi - lo == pytest.approx(0.35, abs=0.08)

    def test_wall_bleeding_through_mask_is_rejected(self):
        chair = surface(2.0, 0.35, n=400)
        wall = surface(3.5, 0.10, n=150)          # 43% of the samples
        _, lo, hi = select_object_depths(np.concatenate([chair, wall]))
        # The old fixed window (4.0 m) kept the wall and reported ~1.5-2.2 m here.
        assert hi - lo < 0.6
        assert hi < 2.5

    def test_centre_tracks_the_object_not_the_mixture(self):
        chair = surface(2.0, 0.35, n=400)
        wall = surface(3.5, 0.10, n=150)
        z, _, _ = select_object_depths(np.concatenate([chair, wall]))
        assert z == pytest.approx(2.0, abs=0.15)

    def test_floor_ramp_connected_to_the_object_is_only_partly_trimmed(self):
        # A receding floor has no depth gap, so gap segmentation cannot see it and
        # only the tail trim bites. This is the weakest case for the method: the
        # extent lands well below the raw 1.2 m span but above the chair's true
        # 0.35 m. Recorded deliberately so the limit is visible rather than
        # discovered later in the field.
        chair = surface(2.0, 0.35, n=400)
        ramp = np.linspace(2.2, 3.2, 40)
        raw = np.concatenate([chair, ramp])
        _, lo, hi = select_object_depths(raw)
        assert (hi - lo) < 0.7 * (raw.max() - raw.min())   # ~40% of the ramp removed
        assert (hi - lo) < 0.9


class TestDrivingScale:
    """CARLA: a 4.5 m vehicle at 15 m, with distant background behind it."""

    def test_vehicle_extent_survives_at_range(self):
        # The car's own surface is far deeper than any indoor object; contiguity
        # must not chop it up just because it is large.
        _, lo, hi = select_object_depths(surface(15.0, 4.5, n=800))
        assert hi - lo == pytest.approx(4.5, abs=0.7)

    def test_distant_background_is_rejected(self):
        car = surface(15.0, 4.5, n=800)
        background = surface(60.0, 2.0, n=300)
        _, lo, hi = select_object_depths(np.concatenate([car, background]))
        assert hi < 25.0

    def test_sky_outliers_do_not_set_the_extent(self):
        car = surface(15.0, 4.5, n=800)
        sky = np.full(50, 900.0)
        z, lo, hi = select_object_depths(np.concatenate([car, sky]))
        assert hi - lo < 6.0
        assert z == pytest.approx(15.0, abs=1.0)

    def test_same_defaults_serve_both_scales(self):
        # No per-scene retuning: one call signature, two scene scales.
        _, ilo, ihi = select_object_depths(
            np.concatenate([surface(2.0, 0.35, n=400), surface(3.5, 0.1, n=150)]))
        _, dlo, dhi = select_object_depths(
            np.concatenate([surface(15.0, 4.5, n=800), surface(60.0, 2.0, n=300)]))
        assert (ihi - ilo) < 0.6
        assert 3.0 < (dhi - dlo) < 6.0


class TestSeedSelection:
    def test_seed_picks_the_far_surface(self):
        near = surface(2.0, 0.2, n=300)
        far = surface(6.0, 0.3, n=300)
        depths = np.concatenate([near, far])
        z, _, _ = select_object_depths(depths, seed_z=6.0)
        assert z == pytest.approx(6.0, abs=0.2)

    def test_seed_picks_the_near_surface(self):
        near = surface(2.0, 0.2, n=300)
        far = surface(6.0, 0.3, n=300)
        depths = np.concatenate([near, far])
        z, _, _ = select_object_depths(depths, seed_z=2.0)
        assert z == pytest.approx(2.0, abs=0.2)

    def test_seed_inside_a_gap_clamps_instead_of_failing(self):
        # A bbox-centre seed can land on a background pixel between two surfaces;
        # that must still yield a box rather than dropping the detection.
        depths = np.concatenate([surface(2.0, 0.2, n=300), surface(6.0, 0.3, n=300)])
        assert select_object_depths(depths, seed_z=4.0) is not None

    def test_default_seed_is_the_median(self):
        depths = np.concatenate([surface(2.0, 0.2, n=500), surface(6.0, 0.3, n=100)])
        z, _, _ = select_object_depths(depths)
        assert z == pytest.approx(2.0, abs=0.2)


class TestTolerance:
    def test_larger_gap_base_merges_nearby_surfaces(self):
        depths = np.concatenate([surface(2.0, 0.1, n=200), surface(2.6, 0.1, n=200)])
        _, _, tight = select_object_depths(depths, seed_z=2.0, gap_base=0.10)
        _, lo, hi = select_object_depths(depths, seed_z=2.0, gap_base=1.0)
        assert (hi - lo) > 0.4          # merged
        assert tight < 2.3              # split

    def test_relative_term_scales_tolerance_with_range(self):
        # Identical geometry at 2 m and 40 m: the relative term keeps the far
        # object's noisier surface from being split.
        sparse_far = np.sort(np.concatenate([
            np.linspace(40.0, 40.5, 30), np.linspace(40.9, 41.4, 30)]))
        _, lo_rel, hi_rel = select_object_depths(sparse_far, gap_rel=0.05)
        _, lo_abs, hi_abs = select_object_depths(sparse_far, gap_rel=0.0)
        assert (hi_rel - lo_rel) > (hi_abs - lo_abs)

    def test_zero_relative_term_is_allowed(self):
        assert select_object_depths(surface(5.0, 0.4), gap_rel=0.0) is not None
