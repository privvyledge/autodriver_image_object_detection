"""Index an mcap rosbag: per-topic message count, duration, observed rate.

Cheap integrity check — reads only the message index, never deserializes.
Usage: python3 bag_index.py <bag_dir>
"""
import sys
from collections import defaultdict

import rosbag2_py


def open_reader(bag_dir):
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_dir, storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    return reader


def main(bag_dir):
    reader = open_reader(bag_dir)
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    count = defaultdict(int)
    t_first, t_last = {}, {}
    bag_first = bag_last = None

    while reader.has_next():
        topic, _data, stamp = reader.read_next()
        count[topic] += 1
        if topic not in t_first:
            t_first[topic] = stamp
        t_last[topic] = stamp
        if bag_first is None:
            bag_first = stamp
        bag_last = stamp

    total = sum(count.values())
    dur = (bag_last - bag_first) / 1e9
    print(f'bag: {bag_dir}')
    print(f'total messages: {total}   duration: {dur:.2f} s\n')
    print(f'{"topic":58} {"count":>7} {"Hz":>7}  type')
    for topic in sorted(count, key=lambda t: -count[t]):
        span = (t_last[topic] - t_first[topic]) / 1e9
        hz = (count[topic] - 1) / span if span > 0 else 0.0
        print(f'{topic:58} {count[topic]:7d} {hz:7.2f}  {types.get(topic, "?")}')


if __name__ == '__main__':
    main(sys.argv[1])
