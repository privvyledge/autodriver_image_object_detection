"""Dump /gosling1/tf_static from the bag and report the camera->lidar chain."""
import sys

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage

TF_STATIC = '/gosling1/tf_static'


def quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main(bag_dir):
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
           rosbag2_py.ConverterOptions('', ''))
    r.set_filter(rosbag2_py.StorageFilter(topics=[TF_STATIC]))

    edges = {}
    while r.has_next():
        _t, data, _s = r.read_next()
        for tr in deserialize_message(data, TFMessage).transforms:
            parent = tr.header.frame_id.lstrip('/')
            child = tr.child_frame_id.lstrip('/')
            t = tr.transform.translation
            q = tr.transform.rotation
            T = np.eye(4)
            T[:3, :3] = quat_to_R(q.x, q.y, q.z, q.w)
            T[:3, 3] = [t.x, t.y, t.z]
            edges[child] = (parent, T)

    print(f'{len(edges)} static edges\n')
    for child, (parent, T) in sorted(edges.items()):
        xyz = T[:3, 3]
        print(f'  {parent:34} -> {child:34}  t=[{xyz[0]:+.4f} {xyz[1]:+.4f} {xyz[2]:+.4f}]')

    np.save('tf_static_edges.npy', np.array(
        [(c, p, T) for c, (p, T) in edges.items()], dtype=object), allow_pickle=True)
    print('\nsaved tf_static_edges.npy')

    def chain_to_root(frame):
        path = []
        seen = set()
        while frame in edges and frame not in seen:
            seen.add(frame)
            parent, _ = edges[frame]
            path.append(f'{frame} <- {parent}')
            frame = parent
        return path, frame

    for f in ('camera_color_optical_frame', 'camera_depth_optical_frame', 'lidar'):
        path, root = chain_to_root(f)
        print(f'\n{f}: root={root}')
        for step in path:
            print(f'   {step}')


if __name__ == '__main__':
    main(sys.argv[1])
