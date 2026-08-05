"""Cross-check depth-projected detection range against the YDLidar X4 scan.

The X4 is the only range sensor in this bag that is independent of the D435i,
so it is the one reference that can expose a shared depth bias. The driver
pointcloud cannot -- it is derived from the same depth stream.

Method: compose the static camera_color_optical_frame -> lidar transform from
tf_static, project each detection's (cx, cy, z_center) into the lidar frame,
and compare its range/bearing against the nearest-in-time scan_filtered.

Geometry caveat, applied as a filter rather than ignored: the lidar plane sits
~0.135 m above the ground, so it strikes an object's base while the detection
centroid sits at its visual centre. Range at a shared bearing is still
comparable for roughly-vertical objects; detections whose bearing falls outside
the scan's valid arc, or where the scan returns inf, are dropped.

Usage: python3 lidar_crosscheck.py <extract.npz> <bag_dir> [--gap-base .1] ...
"""
import argparse
import pathlib
import sys
from collections import defaultdict

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage

REPO = str(pathlib.Path(__file__).resolve().parents[1])  # package repo root
sys.path.insert(0, REPO)
from autodriver_image_object_detection.utils.pointcloud_utils import (  # noqa: E402
    select_object_depths,
)

COLOR = '/gosling1/camera/color/image_raw'
SCAN = '/gosling1/lidar/scan_filtered'
TF_STATIC = '/gosling1/tf_static'
CAM_FRAME = 'camera_color_optical_frame'
LIDAR_FRAME = 'lidar'


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def reader(bag_dir, topics):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return r


def load_static(bag_dir):
    r = reader(bag_dir, [TF_STATIC])
    edges = {}
    while r.has_next():
        _t, data, _s = r.read_next()
        for tr in deserialize_message(data, TFMessage).transforms:
            T = np.eye(4)
            q, t = tr.transform.rotation, tr.transform.translation
            T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [t.x, t.y, t.z]
            edges[tr.child_frame_id.lstrip('/')] = (tr.header.frame_id.lstrip('/'), T)
    return edges


def frame_to_root(edges, frame):
    """T_root_frame: maps a point in `frame` into the tree root."""
    T = np.eye(4)
    seen = set()
    while frame in edges and frame not in seen:
        seen.add(frame)
        parent, E = edges[frame]
        T = E @ T
        frame = parent
    return T, frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('bag_dir')
    ap.add_argument('--gap-base', type=float, default=0.10)
    ap.add_argument('--gap-rel', type=float, default=0.02)
    ap.add_argument('--tail', type=float, default=5.0)
    ap.add_argument('--bearing-win', type=float, default=2.0,
                    help='half-width of the scan bearing window, degrees')
    ap.add_argument('--max-dt', type=float, default=0.12,
                    help='max color/scan time offset, seconds (scan is ~8.7 Hz)')
    ap.add_argument('--max-z', type=float, default=0.35,
                    help='keep only detections whose centroid is within this height '
                         'of the scan plane -- otherwise the beam passes under/over '
                         'the object and returns the wall behind it')
    ap.add_argument('--max-spread', type=float, default=0.30,
                    help='reject a bearing window whose returns are not a coherent '
                         'surface (p90-p10 spread, metres)')
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    rec, off, samp, K = z['records'], z['offsets'], z['samples'], z['K']
    names = z['names']
    fx, fy, cx0, cy0 = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    edges = load_static(args.bag_dir)
    T_root_cam, root_a = frame_to_root(edges, CAM_FRAME)
    T_root_lidar, root_b = frame_to_root(edges, LIDAR_FRAME)
    if root_a != root_b:
        sys.exit(f'frames do not share a root: {root_a} vs {root_b}')
    T_lidar_cam = np.linalg.inv(T_root_lidar) @ T_root_cam
    print(f'T_lidar_cam translation: {T_lidar_cam[:3, 3]}')

    # color frame timestamps + scans
    r = reader(args.bag_dir, [COLOR])
    c_st = []
    while r.has_next():
        _t, _d, s = r.read_next()
        c_st.append(s)
    c_st = np.asarray(c_st, dtype=np.int64)

    r = reader(args.bag_dir, [SCAN])
    scans, s_st = [], []
    while r.has_next():
        _t, data, s = r.read_next()
        scans.append(deserialize_message(data, LaserScan))
        s_st.append(s)
    s_st = np.asarray(s_st, dtype=np.int64)
    print(f'{len(c_st)} color frames, {len(scans)} scans')

    win = np.deg2rad(args.bearing_win)
    rows = []
    per_class = defaultdict(list)
    n_skip_z = n_skip_spread = 0

    for i in range(len(rec)):
        cidx, cls = int(rec[i, 0]), int(rec[i, 1])
        u, v = rec[i, 3], rec[i, 4]
        d = samp[off[i]:off[i + 1]]
        sel = select_object_depths(d, gap_base=args.gap_base, gap_rel=args.gap_rel,
                                   tail_percentile=args.tail)
        if sel is None:
            continue
        zc, zlo, zhi = sel

        # pixel + range -> point in the colour optical frame
        p_cam = np.array([(u - cx0) / fx * zc, (v - cy0) / fy * zc, zc, 1.0])
        p_l = T_lidar_cam @ p_cam
        X, Y, Z = p_l[0], p_l[1], p_l[2]
        rng_cam = float(np.hypot(X, Y))
        bear = float(np.arctan2(Y, X))

        # nearest scan in time
        j = int(np.clip(np.searchsorted(s_st, c_st[cidx]), 0, len(scans) - 1))
        for cand in (j - 1, j):
            if 0 <= cand < len(scans):
                if abs(s_st[cand] - c_st[cidx]) < abs(s_st[j] - c_st[cidx]):
                    j = cand
        if abs(s_st[j] - c_st[cidx]) / 1e9 > args.max_dt:
            continue
        sc = scans[j]
        ang = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment
        rr = np.asarray(sc.ranges, dtype=np.float64)
        m = (np.abs(np.arctan2(np.sin(ang - bear), np.cos(ang - bear))) <= win)
        m &= np.isfinite(rr) & (rr >= sc.range_min) & (rr <= sc.range_max)
        if m.sum() < 3:
            continue
        # Only compare where the beam can actually have struck the object.
        if abs(Z) > args.max_z:
            n_skip_z += 1
            continue
        w = rr[m]
        spread = float(np.percentile(w, 90) - np.percentile(w, 10))
        if spread > args.max_spread:
            n_skip_spread += 1
            continue
        rng_lidar = float(np.median(w))    # coherent surface -> median is the surface
        rows.append((cls, rng_cam, rng_lidar, zc, zhi - zlo, Z, bear))
        per_class[cls].append(rng_cam - rng_lidar)

    if not rows:
        sys.exit('no comparable detections')
    a = np.array(rows)
    err = a[:, 1] - a[:, 2]
    print(f'\nskipped: {n_skip_z} off the scan plane, {n_skip_spread} incoherent window')
    print(f'{len(a)} detections comparable')
    print(f'camera range  median {np.median(a[:, 1]):.2f} m')
    print(f'lidar  range  median {np.median(a[:, 2]):.2f} m')
    print(f'error (cam - lidar): median {np.median(err):+.3f}  '
          f'mean {err.mean():+.3f}  std {err.std():.3f}')
    print(f'  |err| p50 {np.percentile(np.abs(err), 50):.3f}  '
          f'p90 {np.percentile(np.abs(err), 90):.3f}')

    print('\nby range bin (camera):')
    for lo, hi in ((0, 1.5), (1.5, 3), (3, 5), (5, 10)):
        m = (a[:, 1] >= lo) & (a[:, 1] < hi)
        if m.sum() < 5:
            continue
        e = err[m]
        print(f'  {lo:4.1f}-{hi:4.1f} m  n={m.sum():4d}  '
              f'median {np.median(e):+.3f}  |err| p90 {np.percentile(np.abs(e), 90):.3f}')

    print('\nby class (n>=15):')
    for c, v in sorted(per_class.items(), key=lambda t: -len(t[1])):
        if len(v) < 15:
            continue
        v = np.asarray(v)
        print(f'  {names[c]:<16} n={len(v):4d}  median {np.median(v):+.3f}  '
              f'|err| p90 {np.percentile(np.abs(v), 90):.3f}')


if __name__ == '__main__':
    main()
