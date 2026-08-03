"""Put the repository root on sys.path so `autodriver_image_object_detection.*` imports resolve.

This package directory is the repository root, and the Python package lives one
level down in `autodriver_image_object_detection/`. Tests run without a built
workspace, so the path is inserted here rather than relying on `install/`.

Also gates `test/integration/`, which builds real ROS 2 nodes and loads real
weights, behind `--integration` so the default `pytest test/` stays pure-Python
and fast.
"""
import os
import sys

import pytest

TEST_ROOT = os.path.dirname(__file__)
INTEGRATION_DIR = os.path.join(TEST_ROOT, 'integration')

sys.path.insert(0, os.path.abspath(os.path.join(TEST_ROOT, '..')))


def pytest_addoption(parser):
    parser.addoption(
        '--integration', action='store_true', default=False,
        help='also run test/integration — builds real ROS 2 nodes, needs rclpy + weights',
    )


def pytest_configure(config):
    config.addinivalue_line(
        'markers', 'integration: node-level test; needs rclpy, ultralytics and weights')


def pytest_collection_modifyitems(config, items):
    if config.getoption('--integration'):
        return
    # Running them by path is an equally explicit opt-in.
    if all(os.path.abspath(str(arg).split('::')[0]).startswith(INTEGRATION_DIR)
           for arg in config.args):
        return
    skip = pytest.mark.skip(reason='node-level test; pass --integration to run')
    for item in items:
        if str(item.fspath).startswith(INTEGRATION_DIR):
            item.add_marker(skip)
