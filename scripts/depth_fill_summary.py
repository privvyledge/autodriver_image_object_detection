"""Depth-fill and clamp-impact summary for the extracted detections."""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from autodriver_image_object_detection.utils.pointcloud_utils import (  # noqa: E402
    select_object_depths,
)

FLOOR, CAP = 0.2, 4.0

z = np.load(sys.argv[1], allow_pickle=False)
rec, off, samp, names = z['records'], z['offsets'], z['samples'], z['names']
cls = rec[:, 1].astype(int)
mask_px, valid_px = rec[:, 7], rec[:, 8]

print(f'{len(rec)} detections\n')

fill = valid_px / np.maximum(mask_px, 1)
print('depth fill inside the segmentation mask (IR emitter was OFF):')
for q in (10, 25, 50, 75, 90):
    print(f'  p{q:<3} {np.percentile(fill, q) * 100:5.1f}%')
print(f'  detections with <50% fill: {(fill < 0.5).mean() * 100:.1f}%')
print(f'  detections with <10% fill: {(fill < 0.1).mean() * 100:.1f}%')

print('\nfill by class:')
for c in np.unique(cls):
    m = cls == c
    if m.sum() < 20:
        continue
    print(f'  {names[c]:<16} n={m.sum():4d}  median fill {np.median(fill[m]) * 100:5.1f}%')

ext, rng = [], []
for i in range(len(rec)):
    d = samp[off[i]:off[i + 1]]
    out = select_object_depths(d)
    if out is None:
        continue
    zc, lo, hi = out
    ext.append(hi - lo)
    rng.append(zc)
ext, rng = np.asarray(ext), np.asarray(rng)

print(f'\nview-axis extent at current defaults (gap 0.10/0.02, tail 5%):')
print(f'  median {np.median(ext):.3f}  p90 {np.percentile(ext, 90):.3f}  '
      f'max {ext.max():.3f}')
print(f'  below the {FLOOR} m floor: {(ext < FLOOR).mean() * 100:.1f}%  '
      f'-> reported as exactly {FLOOR} m')
print(f'  above the {CAP} m cap:    {(ext > CAP).mean() * 100:.1f}%')

print('\nfloor impact by range:')
for lo_, hi_ in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 10)):
    m = (rng >= lo_) & (rng < hi_)
    if m.sum() < 15:
        continue
    print(f'  {lo_:2d}-{hi_:2d} m  n={m.sum():4d}  median ext {np.median(ext[m]):.3f}  '
          f'floor-clamped {(ext[m] < FLOOR).mean() * 100:5.1f}%')
