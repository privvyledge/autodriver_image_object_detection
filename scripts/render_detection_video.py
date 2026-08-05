"""Render an annotated detection video straight from the bag, no ROS graph.

Why offline: replaying this bag in real time needs ~5x the disk read bandwidth
available on this machine, so the player stalls and publishes intermittently --
and WSLg gives no composited X root to screen-grab. Reading the bag directly
sidesteps both and is deterministic.

Geometry matches the node's depth path: the same select_object_depths() helper
supplies the view-axis extent, the same colour intrinsics unproject the centre,
and the same tf_static chain maps the optical frame into output_frame. Node-level
output was separately confirmed live (Detection3DArray in sensor_kit_link).

Usage: python3 render_detection_video.py <bag_dir> <out.mp4> [--stride 2] ...
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image
from tf2_msgs.msg import TFMessage

REPO = str(pathlib.Path(__file__).resolve().parents[1])  # package repo root
sys.path.insert(0, REPO)
from autodriver_image_object_detection.utils.pointcloud_utils import (  # noqa: E402
    select_object_depths,
)

COLOR = '/gosling1/camera/color/image_raw'
DEPTH = '/gosling1/camera/aligned_depth_to_color/image_raw'
CINFO = '/gosling1/camera/color/camera_info'
TF_STATIC = '/gosling1/tf_static'
DEPTH_SCALE = 1000.0
DEPTH_MAX = 10.0
MIN_DEPTH_THICKNESS = 0.2
DEPTH_BOX_THICKNESS = 4.0
CAM_FRAME = 'camera_color_optical_frame'
OUT_FRAME = 'sensor_kit_link'
PALETTE = {'person': (0, 200, 255), 'car': (255, 128, 0), 'chair': (0, 255, 120)}


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


def static_chain(bag_dir, child, ancestor):
    r = reader(bag_dir, [TF_STATIC])
    edges = {}
    while r.has_next():
        _t, d, _s = r.read_next()
        for tr in deserialize_message(d, TFMessage).transforms:
            T = np.eye(4)
            q, t = tr.transform.rotation, tr.transform.translation
            T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [t.x, t.y, t.z]
            edges[tr.child_frame_id.lstrip('/')] = (tr.header.frame_id.lstrip('/'), T)

    def to_root(f):
        T, seen = np.eye(4), set()
        while f in edges and f not in seen:
            seen.add(f)
            p, E = edges[f]
            T = E @ T
            f = p
        return T, f
    Tc, _ = to_root(child)
    Ta, _ = to_root(ancestor)
    return np.linalg.inv(Ta) @ Tc


def decode(msg):
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('16UC1', 'mono16'):
        return np.ascontiguousarray(
            buf.view(np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width])
    a = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(
        msg.height, msg.width, 3)
    return np.ascontiguousarray(a[:, :, ::-1] if msg.encoding == 'rgb8' else a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag_dir')
    ap.add_argument('out')
    ap.add_argument('--stride', type=int, default=2)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--conf', type=float, default=0.3)
    ap.add_argument('--model', default=f'{REPO}/yolo11n-seg.pt')
    args = ap.parse_args()

    T_out_cam = static_chain(args.bag_dir, CAM_FRAME, OUT_FRAME)
    print(f'T_{OUT_FRAME}_{CAM_FRAME} translation {np.round(T_out_cam[:3, 3], 4)}')

    # index + pair
    r = reader(args.bag_dir, [COLOR, DEPTH])
    cs, ds = [], []
    while r.has_next():
        t, _d, s = r.read_next()
        (cs if t == COLOR else ds).append(s)
    cs, ds = np.array(cs, np.int64), np.array(ds, np.int64)
    want = np.arange(0, len(cs), args.stride)
    pos = np.clip(np.searchsorted(ds, cs[want]), 1, len(ds) - 1)
    pick = np.where(np.abs(ds[pos - 1] - cs[want]) < np.abs(ds[pos] - cs[want]),
                    pos - 1, pos)
    c2d = {int(c): int(d) for c, d in zip(want, pick)}
    need_d = set(c2d.values())
    want_c = set(int(x) for x in want)

    from ultralytics import YOLO
    model = YOLO(args.model)
    names = model.names
    keep = [i for i, n in names.items() if n in PALETTE]
    print(f'classes: {[names[i] for i in keep]}')

    r = reader(args.bag_dir, [COLOR, DEPTH, CINFO])
    K = None
    dcache, pend = {}, {}
    ci = cj = 0
    writer = None
    nframes = ndet = 0

    def process(idx):
        nonlocal writer, nframes, ndet
        img = pend.pop(idx)
        depth = dcache[c2d[idx]].astype(np.float32) / DEPTH_SCALE
        res = model.predict(img, conf=args.conf, classes=keep, verbose=False,
                            retina_masks=True)[0]
        vis = img.copy()
        lines = []
        if res.masks is not None and len(res.boxes):
            masks = res.masks.data.cpu().numpy().astype(bool)
            xyxy = res.boxes.xyxy.cpu().numpy()
            xywh = res.boxes.xywh.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            cf = res.boxes.conf.cpu().numpy()
            for k in range(len(cls)):
                lab = names[cls[k]]
                col = PALETTE.get(lab, (200, 200, 200))
                m = masks[k]
                if m.shape == vis.shape[:2]:
                    vis[m] = (0.6 * vis[m] + 0.4 * np.array(col)).astype(np.uint8)
                x1, y1, x2, y2 = xyxy[k].astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)

                txt = f'{lab} {cf[k]:.2f}'
                if m.shape == depth.shape:
                    d = depth[m]
                    d = d[np.isfinite(d) & (d > 0) & (d <= DEPTH_MAX)]
                    if d.size:
                        sel = select_object_depths(d)
                        if sel:
                            z, lo, hi = sel
                            ext = float(min(max(hi - lo, MIN_DEPTH_THICKNESS),
                                            DEPTH_BOX_THICKNESS))
                            u, v = xywh[k][0], xywh[k][1]
                            p = np.array([(u - K[0, 2]) / K[0, 0] * z,
                                          (v - K[1, 2]) / K[1, 1] * z, z, 1.0])
                            q = T_out_cam @ p
                            sx = z * xywh[k][2] / K[0, 0]
                            sy = z * xywh[k][3] / K[1, 1]
                            txt += f'  {z:.2f} m'
                            lines.append(f'{lab}: xyz {q[0]:+.2f} {q[1]:+.2f} '
                                         f'{q[2]:+.2f} m  size {ext:.2f} x {sx:.2f} '
                                         f'x {sy:.2f}')
                            ndet += 1
                cv2.putText(vis, txt, (x1, max(y1 - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

        panel = np.zeros((104, vis.shape[1], 3), np.uint8)
        cv2.putText(panel, f'3D boxes in {OUT_FRAME}   (frame {idx})', (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for i, t in enumerate(lines[:4]):
            cv2.putText(panel, t, (8, 42 + i * 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, (255, 255, 255), 1)
        if not lines:
            cv2.putText(panel, '(no detection with valid depth)', (8, 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (150, 150, 150), 1)
        frame = np.vstack([vis, panel])
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                     args.fps, (w, h))
            print(f'writing {w}x{h} -> {args.out}')
        writer.write(frame)
        nframes += 1
        if nframes % 50 == 0:
            print(f'  {nframes} frames, {ndet} 3D dets', flush=True)

    while r.has_next():
        t, data, _s = r.read_next()
        if t == CINFO:
            if K is None:
                K = np.asarray(deserialize_message(data, CameraInfo).k).reshape(3, 3)
            continue
        if t == DEPTH:
            if cj in need_d:
                dcache[cj] = decode(deserialize_message(data, Image))
            cj += 1
        else:
            if ci in want_c:
                pend[ci] = decode(deserialize_message(data, Image))
            ci += 1
        for idx in [c for c in pend if c2d[c] in dcache and K is not None]:
            process(idx)

    if writer:
        writer.release()
    print(f'done: {nframes} frames, {ndet} 3D detections -> {args.out}')


if __name__ == '__main__':
    main()
