"""Validate the camera->lidar extrinsic and depth scale with no YOLO involved.

Takes the driver's XYZRGB cloud (camera_depth_optical_frame), transforms it into
the lidar frame, slices a thin horizontal band around the scan plane, and
compares range-vs-bearing against the time-nearest scan_filtered.

If the depth stream and the extrinsic are both right, the two range profiles
overlay. Any constant offset is an extrinsic error; a multiplicative one is a
depth-scale error. This has to pass before a detection-level comparison against
the lidar means anything.
"""
import argparse
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from tf2_msgs.msg import TFMessage

CLOUD = '/gosling1/camera/depth/color/points'
SCAN = '/gosling1/lidar/scan_filtered'
TF_STATIC = '/gosling1/tf_static'


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


def to_root(edges, frame):
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
    ap.add_argument('bag_dir')
    ap.add_argument('--frames', type=int, default=5)
    ap.add_argument('--band', type=float, default=0.03, help='slice half-thickness (m)')
    args = ap.parse_args()

    edges = load_static(args.bag_dir)
    T_rc, ra = to_root(edges, 'camera_depth_optical_frame')
    T_rl, rb = to_root(edges, 'lidar')
    assert ra == rb, (ra, rb)
    T_lc = np.linalg.inv(T_rl) @ T_rc
    print(f'T_lidar_cam:\n{np.round(T_lc, 4)}\n')

    # collect scans first
    r = reader(args.bag_dir, [SCAN])
    scans, s_st = [], []
    while r.has_next():
        _t, d, s = r.read_next()
        scans.append(deserialize_message(d, LaserScan))
        s_st.append(s)
    s_st = np.asarray(s_st, dtype=np.int64)

    r = reader(args.bag_dir, [CLOUD])
    n_done = 0
    idx = 0
    step = max(1, 1700 // max(args.frames, 1))
    while r.has_next() and n_done < args.frames:
        _t, data, stamp = r.read_next()
        idx += 1
        if idx % step:
            continue
        msg = deserialize_message(data, PointCloud2)
        pts = pc2.read_points_numpy(msg, field_names=('x', 'y', 'z'), skip_nans=True)
        if pts.size == 0:
            continue
        P = np.c_[pts, np.ones(len(pts))] @ T_lc.T
        band = np.abs(P[:, 2]) <= args.band
        Pb = P[band]
        if len(Pb) < 200:
            print(f'frame {idx}: only {len(Pb)} pts in band, skipping')
            continue

        j = int(np.clip(np.searchsorted(s_st, stamp), 0, len(scans) - 1))
        sc = scans[j]
        dt = abs(s_st[j] - stamp) / 1e9

        bear_c = np.arctan2(Pb[:, 1], Pb[:, 0])
        rng_c = np.hypot(Pb[:, 0], Pb[:, 1])
        ang = sc.angle_min + np.arange(len(sc.ranges)) * sc.angle_increment
        rr = np.asarray(sc.ranges, dtype=np.float64)
        good = np.isfinite(rr) & (rr >= sc.range_min) & (rr <= sc.range_max)

        # compare in 1-degree bearing bins where both have data
        bins = np.deg2rad(np.arange(-30, 31, 1.0))
        diffs = []
        for b0, b1 in zip(bins[:-1], bins[1:]):
            mc = (bear_c >= b0) & (bear_c < b1)
            ml = good & (ang >= b0) & (ang < b1)
            if mc.sum() < 5 or ml.sum() < 1:
                continue
            diffs.append(np.median(rng_c[mc]) - np.median(rr[ml]))
        if not diffs:
            print(f'frame {idx}: no overlapping bearing bins')
            continue
        d = np.asarray(diffs)
        print(f'frame {idx:5d}  dt={dt * 1e3:5.1f}ms  band_pts={len(Pb):6d}  '
              f'bins={len(d):3d}  median(cloud-scan)={np.median(d):+.3f} m  '
              f'IQR={np.percentile(d, 75) - np.percentile(d, 25):.3f}')
        n_done += 1


if __name__ == '__main__':
    main()
