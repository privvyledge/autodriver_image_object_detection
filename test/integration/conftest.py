"""Fixtures for the node-level tests.

Unlike `test/`, these construct *real* ROS 2 nodes: they need `rclpy`, `ultralytics`
and a `.pt` weights file, and they take a few seconds each. They are skipped —
never failed — when any of that is missing, so `python3 -m pytest test/ -q` stays
green and fast on a machine without a ROS install.

Each test gets its own rclpy context. Node classes here take no constructor
arguments and do not forward `parameter_overrides`, so the only way to influence a
node's parameters is the global `--ros-args -p name:=value` set at `rclpy.init()`
time — hence one init/shutdown per test rather than a session-scoped context.
"""
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
MODEL_PATH = os.path.join(REPO_ROOT, 'yolo11n-seg.pt')

try:
    import rclpy  # noqa: F401
    _MISSING = None
except ImportError as e:  # pragma: no cover - environment dependent
    _MISSING = f'rclpy not importable: {e}'

if _MISSING is None:
    try:
        import ultralytics  # noqa: F401
    except ImportError as e:  # pragma: no cover - environment dependent
        _MISSING = f'ultralytics not importable: {e}'


def _fmt(value):
    """Render a Python value the way the ROS 2 `-p name:=value` CLI expects."""
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple)):
        return '[' + ','.join(str(v) for v in value) + ']'
    return str(value)


@pytest.fixture
def make_node():
    """Yield ``make(node_cls, **param_overrides) -> node``.

    Brings up a fresh rclpy context with the given parameter overrides, builds the
    node, and tears both down afterwards. One node per test.
    """
    if _MISSING is not None:
        pytest.skip(_MISSING)

    import rclpy

    state = {'inited': False, 'node': None}

    def make(node_cls, **overrides):
        assert state['node'] is None, 'make_node builds one node per test'
        args = ['--ros-args']
        for key, value in overrides.items():
            args += ['-p', f'{key}:={_fmt(value)}']
        rclpy.init(args=args)
        state['inited'] = True
        state['node'] = node_cls()
        return state['node']

    yield make

    node = state['node']
    if node is not None:
        # A TransformListener(spin_thread=True) owns a background executor that raises
        # ExternalShutdownException out of its thread if rclpy.shutdown() beats it.
        listener = getattr(node, 'tf_listener', None)
        executor = getattr(listener, 'executor', None)
        if executor is not None:
            executor.shutdown()
        node.destroy_node()
    if state['inited']:
        rclpy.shutdown()


@pytest.fixture
def model_path():
    """Absolute path to the .pt weights the node tests load."""
    if _MISSING is not None:
        pytest.skip(_MISSING)
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f'{MODEL_PATH} not present')
    return MODEL_PATH
