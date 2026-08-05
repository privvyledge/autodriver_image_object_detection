"""Record what the detector actually publishes: class labels, 2D pixels, 3D box geometry.

Answers the question a message count cannot -- whether the 3D boxes are sane
(plausible centre, non-degenerate size, correct frame) rather than merely present.

Usage: python3 inspect_detections.py <seconds> <out.json>
"""
import json
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray, Detection3DArray

COCO = {0: 'person', 2: 'car', 56: 'chair'}


def label_of(det):
    """Ultralytics writes the class id into results[0].hypothesis.class_id as a string."""
    if not det.results:
        return '?'
    cid = det.results[0].hypothesis.class_id
    try:
        return COCO.get(int(cid), str(cid))
    except (TypeError, ValueError):
        return str(cid)


class Inspector(Node):
    def __init__(self):
        super().__init__('detection_inspector')
        q = QoSProfile(depth=50)
        q.reliability = ReliabilityPolicy.BEST_EFFORT
        self.rows2, self.rows3 = [], []
        self.frames2 = self.frames3 = 0
        self.create_subscription(Detection2DArray, '/yolo/detection_results', self.cb2, q)
        self.create_subscription(Detection3DArray, '/yolo/detection3d_depth_results',
                                 self.cb3, q)

    def cb2(self, m):
        self.frames2 += 1
        for d in m.detections:
            self.rows2.append({
                'frame': m.header.frame_id,
                'label': label_of(d),
                'score': float(d.results[0].hypothesis.score) if d.results else None,
                'cx': float(d.bbox.center.position.x),
                'cy': float(d.bbox.center.position.y),
                'w': float(d.bbox.size_x), 'h': float(d.bbox.size_y),
                'track_id': d.id,
            })

    def cb3(self, m):
        self.frames3 += 1
        for d in m.detections:
            c = d.bbox.center.position
            s = d.bbox.size
            self.rows3.append({
                'frame': m.header.frame_id,
                'label': label_of(d),
                'x': float(c.x), 'y': float(c.y), 'z': float(c.z),
                'sx': float(s.x), 'sy': float(s.y), 'sz': float(s.z),
            })


def main():
    dur = float(sys.argv[1])
    out = sys.argv[2]
    rclpy.init()
    n = Inspector()
    t0 = time.time()
    while time.time() - t0 < dur:
        rclpy.spin_once(n, timeout_sec=0.2)

    by2 = defaultdict(int)
    for r in n.rows2:
        by2[r['label']] += 1
    print(f'2D: {n.frames2} msgs, {len(n.rows2)} detections')
    for k, v in sorted(by2.items(), key=lambda t: -t[1]):
        print(f'   {k:<10} {v}')
    print(f'3D: {n.frames3} msgs, {len(n.rows3)} detections')
    if n.rows3:
        frames = {r['frame'] for r in n.rows3}
        print(f'   output frame(s): {frames}')
        by3 = defaultdict(list)
        for r in n.rows3:
            by3[r['label']].append(r)
        for k, v in sorted(by3.items(), key=lambda t: -len(t[1])):
            import statistics as st
            print(f'   {k:<10} n={len(v):4d}  '
                  f'x {st.median([r["x"] for r in v]):6.2f}  '
                  f'y {st.median([r["y"] for r in v]):6.2f}  '
                  f'z {st.median([r["z"] for r in v]):6.2f} | '
                  f'size {st.median([r["sx"] for r in v]):.2f} x '
                  f'{st.median([r["sy"] for r in v]):.2f} x '
                  f'{st.median([r["sz"] for r in v]):.2f} m')
        degen = [r for r in n.rows3 if min(r['sx'], r['sy'], r['sz']) <= 0.0]
        print(f'   degenerate boxes (any size <= 0): {len(degen)}')

    with open(out, 'w') as f:
        json.dump({'d2': n.rows2, 'd3': n.rows3,
                   'frames2': n.frames2, 'frames3': n.frames3}, f)
    print(f'wrote {out}')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
