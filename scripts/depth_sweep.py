"""Score select_object_depths() parameter settings against extracted depth samples.

Reads the .npz from depth_extract.py and replays the real helper (imported from
the package, not a copy) over every detection's raw in-mask depth samples for a
grid of (gap_base, gap_rel, tail_percentile). Fully deterministic: no inference,
no ROS graph, no bag replay.

Reported per setting:
  ext_med / ext_p90   view-axis extent distribution (metres)
  floor% / cap%       share clamped by MIN_DEPTH_THICKNESS / depth_box_thickness
  seg%                share where gap segmentation actually dropped samples
  kept%               mean share of samples surviving both filters
  dz_med              median |z_center - raw median|, i.e. how far the filter
                      moves the range estimate away from the naive answer

Usage: python3 depth_sweep.py <extract.npz> [--by-class] [--range-bins]
"""
import argparse
import os
import pathlib
import sys

import numpy as np

REPO = str(pathlib.Path(__file__).resolve().parents[1])  # package repo root
sys.path.insert(0, REPO)
from autodriver_image_object_detection.utils.pointcloud_utils import (  # noqa: E402
    select_object_depths,
)

MIN_DEPTH_THICKNESS = 0.2   # yolo_detection_node floor
DEPTH_BOX_THICKNESS = 4.0   # yolo_detection_node last-resort cap


def load(path):
    z = np.load(path, allow_pickle=False)
    rec, off, samp = z['records'], z['offsets'], z['samples']
    names = z['names']
    dets = [samp[off[i]:off[i + 1]] for i in range(len(rec))]
    return rec, dets, names


def score(dets, gap_base, gap_rel, tail):
    ext, dz, kept, seg = [], [], [], 0
    for d in dets:
        raw_med = float(np.median(d))
        out = select_object_depths(d, gap_base=gap_base, gap_rel=gap_rel,
                                   tail_percentile=tail)
        if out is None:
            continue
        z, lo, hi = out
        ext.append(hi - lo)
        dz.append(abs(z - raw_med))
        # how many samples survive the same gap step the helper applies
        order = np.sort(np.asarray(d, dtype=np.float64))
        tol = gap_base + gap_rel * max(raw_med, 0.0)
        breaks = np.flatnonzero(np.diff(order) > tol) + 1
        if breaks.size:
            starts = np.concatenate(([0], breaks))
            ends = np.concatenate((breaks, [order.size]))
            idx = int(np.clip(np.searchsorted(order, raw_med, side='right') - 1,
                              0, order.size - 1))
            run = int(np.searchsorted(ends, idx, side='right'))
            n_run = ends[run] - starts[run]
            seg += int(n_run < order.size)
            kept.append(n_run / order.size)
        else:
            kept.append(1.0)
    ext = np.asarray(ext)
    clamped = np.clip(ext, MIN_DEPTH_THICKNESS, DEPTH_BOX_THICKNESS)
    return {
        'n': len(ext),
        'ext_med': float(np.median(ext)),
        'ext_p90': float(np.percentile(ext, 90)),
        'floor%': 100.0 * float((ext < MIN_DEPTH_THICKNESS).mean()),
        'cap%': 100.0 * float((ext > DEPTH_BOX_THICKNESS).mean()),
        'seg%': 100.0 * seg / max(len(dets), 1),
        'kept%': 100.0 * float(np.mean(kept)),
        'dz_med': float(np.median(dz)),
        'clamped_med': float(np.median(clamped)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('npz')
    ap.add_argument('--by-class', action='store_true')
    args = ap.parse_args()

    rec, dets, names = load(args.npz)
    print(f'{len(dets)} detections from {args.npz}\n')

    cls = rec[:, 1].astype(int)
    uniq, cnt = np.unique(cls, return_counts=True)
    print('classes present:')
    for c, n in sorted(zip(uniq, cnt), key=lambda t: -t[1]):
        med = np.median([np.median(d) for d, k in zip(dets, cls) if k == c])
        print(f'  {names[c]:<16} {n:5d}   median range {med:.2f} m')
    print()

    grid = []
    for gb in (0.05, 0.10, 0.15, 0.20, 0.30):
        grid.append((gb, 0.02, 5.0))
    for gr in (0.0, 0.01, 0.04, 0.08):
        grid.append((0.10, gr, 5.0))
    for tp in (0.0, 2.5, 10.0, 15.0):
        grid.append((0.10, 0.02, tp))

    seen = set()
    hdr = (f'{"gap_base":>8} {"gap_rel":>7} {"tail":>5} | {"ext_med":>7} '
           f'{"ext_p90":>7} {"floor%":>6} {"cap%":>5} {"seg%":>5} '
           f'{"kept%":>6} {"dz_med":>6}')
    print(hdr)
    print('-' * len(hdr))
    for gb, gr, tp in grid:
        if (gb, gr, tp) in seen:
            continue
        seen.add((gb, gr, tp))
        s = score(dets, gb, gr, tp)
        star = ' <- current default' if (gb, gr, tp) == (0.10, 0.02, 5.0) else ''
        print(f'{gb:8.2f} {gr:7.2f} {tp:5.1f} | {s["ext_med"]:7.3f} '
              f'{s["ext_p90"]:7.3f} {s["floor%"]:6.1f} {s["cap%"]:5.1f} '
              f'{s["seg%"]:5.1f} {s["kept%"]:6.1f} {s["dz_med"]:6.3f}{star}')

    # naive baseline: no filtering at all
    raw_ext = np.array([d.max() - d.min() for d in dets])
    print(f'\nno filtering (raw min/max): ext_med {np.median(raw_ext):.3f}  '
          f'p90 {np.percentile(raw_ext, 90):.3f}')

    if args.by_class:
        print('\nper-class extent at the current default:')
        for c, n in sorted(zip(uniq, cnt), key=lambda t: -t[1]):
            if n < 20:
                continue
            sub = [d for d, k in zip(dets, cls) if k == c]
            s = score(sub, 0.10, 0.02, 5.0)
            print(f'  {names[c]:<16} n={n:4d}  ext_med {s["ext_med"]:.3f}  '
                  f'p90 {s["ext_p90"]:.3f}  floor% {s["floor%"]:.1f}')


if __name__ == '__main__':
    main()
