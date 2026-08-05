"""Offline A/B of the depth-path view-axis extent, old logic vs new, on the gosling1 bag.

Only the extent is compared, which is frame-independent, so no TF or ROS node is
needed -- just aligned colour+depth pairs and the same YOLO masks the node would use.
"""
import sys
sys.path.insert(0, "/mnt/c/Users/boluo/OneDrive - Florida State University/Projects/"
                   "autodriver/autodriver_perception/autodriver_image_object_detection")

import numpy as np
import cv2
from rosbags.highlevel import AnyReader
from pathlib import Path
from ultralytics import YOLO

from autodriver_image_object_detection.utils.pointcloud_utils import select_object_depths

DEPTH_SCALE = 1000.0
DEPTH_MAX = 6.0
BOX_THICKNESS = 4.0
MIN_THICK = 0.2
COLOR_T = '/gosling1/camera/color/image_raw'
DEPTH_T = '/gosling1/camera/aligned_depth_to_color/image_raw'


def old_extent(depths, seed):
    """Pre-fix logic: fixed +/- thickness/2 window, then min/max."""
    keep = depths[np.abs(depths - seed) <= BOX_THICKNESS / 2.0]
    if keep.size == 0:
        return None
    return float(keep.max() - keep.min())


def new_extent(depths, seed):
    sel = select_object_depths(depths, seed_z=seed)
    if sel is None:
        return None
    _, lo, hi = sel
    return float(hi - lo)   # raw, pre-clamp


def img(msg):
    a = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
    return a


def main():
    model = YOLO('yolo11m-seg.pt')
    colors, depths_msgs = [], []
    with AnyReader([Path.home() / 'bags']) as reader:
        conns = [c for c in reader.connections if c.topic in (COLOR_T, DEPTH_T)]
        for conn, t, raw in reader.messages(connections=conns):
            m = reader.deserialize(raw, conn.msgtype)
            key = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
            (colors if conn.topic == COLOR_T else depths_msgs).append((key, m))

    # approximate sync, mirroring the node's ApproximateTimeSynchronizer
    dkeys = np.array([k for k, _ in depths_msgs])
    pairs = []
    for ck, cm in colors:
        j = int(np.argmin(np.abs(dkeys - ck)))
        if abs(int(dkeys[j]) - ck) < 50_000_000:      # 50 ms
            pairs.append((cm, depths_msgs[j][1]))
    pairs = pairs[:200]

    print(f"{len(pairs)} synchronized colour+depth pairs")
    olds, news, cls_seen = [], [], set()
    for cmsg, dmsg in pairs:
        bgr = cv2.cvtColor(img(cmsg).reshape(cmsg.height, cmsg.width, 3), cv2.COLOR_RGB2BGR)
        depth = np.frombuffer(dmsg.data, dtype=np.uint16).reshape(dmsg.height, dmsg.width)
        res = model.predict(bgr, conf=0.35, verbose=False)[0]
        names = res.names
        if res.masks is None:
            continue
        for mk in res.masks:
            md = (mk.data.cpu().numpy().astype(np.uint8)[0] * 255)
            if md.shape != depth.shape:
                md = cv2.resize(md, (depth.shape[1], depth.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
            roi = depth.copy()
            roi[md == 0] = 0
            roi_m = roi / DEPTH_SCALE
            valid = roi_m[np.isfinite(roi_m) & (roi_m > 0) & (roi_m <= DEPTH_MAX)]
            if valid.size == 0:
                continue
            seed = float(np.median(valid))
            o, n = old_extent(valid, seed), new_extent(valid, seed)
            if o is not None and n is not None:
                olds.append(o)
                news.append(n)
                cls_seen.add(names[int(res.boxes.cls[len(olds)-1])] if len(olds)-1 < len(res.boxes.cls) else '?')

    olds, news = np.array(olds), np.array(news)
    print(f"\ndetections compared: {len(olds)}")
    for name, arr in (("OLD (fixed window + min/max)", olds), ("NEW (gap segmentation)", news)):
        print(f"{name:34s} median={np.median(arr):.2f} m  "
              f"p90={np.percentile(arr, 90):.2f}  max={arr.max():.2f}")
    print("\nNEW is RAW (pre-clamp). classes seen:", cls_seen)
    print(f"fraction that would hit the 0.2 m floor: {(news < 0.2).mean():.0%}")


main()
