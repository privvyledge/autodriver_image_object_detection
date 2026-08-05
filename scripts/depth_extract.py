"""Extract per-detection depth samples from the stationary-obstacle bag.

Two passes so memory stays flat regardless of bag size:
  Pass A  index message timestamps only (no deserialization)
  Pass B  deserialize a subsampled set of color frames + their nearest aligned
          depth frame, run YOLO-seg, and dump the raw in-mask depth samples

The output .npz holds raw depth samples per detection, so any number of
select_object_depths() parameter settings can be scored afterwards without
re-running inference. That keeps the sweep deterministic and cheap.

Usage:
  python3 depth_extract.py <bag_dir> <out.npz> [--stride 6] [--model yolo11n-seg.pt]
"""
import argparse
import sys
from collections import defaultdict

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CameraInfo, Image

COLOR = '/gosling1/camera/color/image_raw'
DEPTH = '/gosling1/camera/aligned_depth_to_color/image_raw'
CINFO = '/gosling1/camera/color/camera_info'
DEPTH_SCALE = 1000.0  # D435i 16UC1 millimetres -> metres


def reader_for(bag_dir, topics=None):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    if topics:
        r.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))
    return r


def index_stamps(bag_dir):
    """Pass A: {topic: [bag_stamp_ns, ...]} for the image topics."""
    r = reader_for(bag_dir, [COLOR, DEPTH])
    stamps = defaultdict(list)
    while r.has_next():
        topic, _data, stamp = r.read_next()
        stamps[topic].append(stamp)
    return {k: np.asarray(v, dtype=np.int64) for k, v in stamps.items()}


def imgmsg_to_array(msg):
    """Decode sensor_msgs/Image without cv_bridge (avoids a numpy ABI dependency)."""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('16UC1', 'mono16'):
        arr = buf.view(np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
    elif msg.encoding in ('rgb8', 'bgr8'):
        arr = buf.reshape(msg.height, msg.step)[:, :msg.width * 3]
        arr = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == 'rgb8':
            arr = arr[:, :, ::-1]  # -> BGR for Ultralytics/OpenCV
    elif msg.encoding == 'mono8':
        arr = buf.reshape(msg.height, msg.step)[:, :msg.width]
    else:
        raise ValueError(f'unhandled encoding {msg.encoding}')
    return np.ascontiguousarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('bag_dir')
    ap.add_argument('out')
    ap.add_argument('--stride', type=int, default=6, help='use every Nth color frame')
    ap.add_argument('--model', default='yolo11n-seg.pt')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--max-pair-dt', type=float, default=0.020,
                    help='reject a color/depth pair further apart than this (s)')
    args = ap.parse_args()

    print('pass A: indexing timestamps ...', flush=True)
    stamps = index_stamps(args.bag_dir)
    c_st, d_st = stamps.get(COLOR), stamps.get(DEPTH)
    if c_st is None or d_st is None:
        sys.exit('missing color or depth topic')
    print(f'  color {len(c_st)}  depth {len(d_st)}')

    # Nearest-depth pairing on bag receive time, decided up front.
    want_c = np.arange(0, len(c_st), args.stride)
    pos = np.searchsorted(d_st, c_st[want_c])
    lo = np.clip(pos - 1, 0, len(d_st) - 1)
    hi = np.clip(pos, 0, len(d_st) - 1)
    pick = np.where(np.abs(d_st[lo] - c_st[want_c]) <= np.abs(d_st[hi] - c_st[want_c]), lo, hi)
    dt = np.abs(d_st[pick] - c_st[want_c]) / 1e9
    ok = dt <= args.max_pair_dt
    want_c, want_d = want_c[ok], pick[ok]
    print(f'  paired {ok.sum()}/{len(ok)} frames  (median dt {np.median(dt[ok]) * 1e3:.1f} ms)')

    want_c_set = {int(i) for i in want_c}
    depth_needed = {int(i) for i in want_d}
    c_to_d = {int(c): int(d) for c, d in zip(want_c, want_d)}

    from ultralytics import YOLO
    model = YOLO(args.model)

    print('pass B: decoding + inference ...', flush=True)
    r = reader_for(args.bag_dir, [COLOR, DEPTH, CINFO])
    depth_cache, pending, K = {}, {}, None
    ci = cj = 0
    records, samples = [], []

    def flush(cidx):
        """Run YOLO on color frame cidx once its depth partner is available."""
        color = pending.pop(cidx)
        depth = depth_cache[c_to_d[cidx]]
        res = model.predict(color, conf=args.conf, verbose=False, retina_masks=True)[0]
        if res.masks is None or len(res.boxes) == 0:
            return
        depth_m = depth.astype(np.float32) / DEPTH_SCALE
        masks = res.masks.data.cpu().numpy().astype(bool)
        boxes = res.boxes
        xywh = boxes.xywh.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        conf = boxes.conf.cpu().numpy()
        for k in range(len(cls)):
            m = masks[k]
            if m.shape != depth_m.shape:
                continue
            d = depth_m[m]
            d = d[np.isfinite(d) & (d > 0)]
            if d.size < 10:
                continue
            records.append((cidx, cls[k], conf[k], xywh[k][0], xywh[k][1],
                            xywh[k][2], xywh[k][3], int(m.sum()), d.size,
                            len(samples)))
            samples.append(d.astype(np.float32))

    while r.has_next():
        topic, data, _stamp = r.read_next()
        if topic == CINFO:
            if K is None:
                K = np.asarray(deserialize_message(data, CameraInfo).k).reshape(3, 3)
            continue
        if topic == DEPTH:
            if cj in depth_needed:
                depth_cache[cj] = imgmsg_to_array(deserialize_message(data, Image))
            cj += 1
        elif topic == COLOR:
            if ci in want_c_set:
                pending[ci] = imgmsg_to_array(deserialize_message(data, Image))
            ci += 1
        # flush any color frame whose depth partner has now been read
        for cidx in [c for c in pending if c_to_d[c] in depth_cache]:
            flush(cidx)
            if len(records) and len(records) % 200 == 0:
                print(f'  {len(records)} detections ...', flush=True)

    rec = np.array(records, dtype=np.float64)
    lens = np.array([s.size for s in samples], dtype=np.int64)
    np.savez_compressed(
        args.out,
        records=rec,
        columns=np.array(['color_idx', 'cls', 'conf', 'cx', 'cy', 'w', 'h',
                          'mask_px', 'valid_px', 'sample_idx']),
        samples=np.concatenate(samples) if samples else np.zeros(0, np.float32),
        offsets=np.concatenate(([0], np.cumsum(lens))),
        names=np.array([model.names[i] for i in range(len(model.names))]),
        K=K if K is not None else np.zeros((3, 3)),
    )
    print(f'wrote {args.out}: {len(records)} detections, '
          f'{sum(lens)} depth samples')


if __name__ == '__main__':
    main()
