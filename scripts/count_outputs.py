"""Count messages and detections on the yolo output topics for N seconds."""
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from vision_msgs.msg import Detection2DArray, Detection3DArray


class Counter(Node):
    def __init__(self):
        super().__init__('yolo_output_counter')
        q = QoSProfile(depth=50)
        q.reliability = ReliabilityPolicy.BEST_EFFORT
        self.n2 = self.d2 = self.n3 = self.d3 = 0
        self.create_subscription(Detection2DArray, '/yolo/detection_results',
                                 self.cb2, q)
        self.create_subscription(Detection3DArray, '/yolo/detection3d_depth_results',
                                 self.cb3, q)

    def cb2(self, m):
        self.n2 += 1
        self.d2 += len(m.detections)

    def cb3(self, m):
        self.n3 += 1
        self.d3 += len(m.detections)


def main():
    dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
    rclpy.init()
    n = Counter()
    t0 = time.time()
    last = 0
    while time.time() - t0 < dur:
        rclpy.spin_once(n, timeout_sec=0.2)
        el = int(time.time() - t0)
        if el != last and el % 10 == 0:
            print(f'  t={el:3d}s  2D msgs={n.n2} dets={n.d2}  '
                  f'3D msgs={n.n3} dets={n.d3}', flush=True)
            last = el
    print(f'TOTAL over {dur:.0f}s: Detection2DArray {n.n2} msgs / {n.d2} dets, '
          f'Detection3DArray {n.n3} msgs / {n.d3} dets')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
