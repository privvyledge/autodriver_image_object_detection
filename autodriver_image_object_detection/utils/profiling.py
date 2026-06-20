"""Optional per-stage execution-time profiling for detection nodes.

Times named pipeline stages (e.g. detection inference, depth/pointcloud projection)
*in-node*, independent of whether a downstream message is actually published. This is
the reliable way to gauge compute cost when nodes only publish on fresh/valid data and
`ros2 topic hz` cannot distinguish "slow" from "nothing to publish this frame".

Everything is off by default and gated behind ROS2 parameters. When disabled, the
`measure()` context manager is a near-zero no-op.

ROS imports are kept lazy inside functions so this module stays importable from
non-ROS scripts.
"""
import time
from collections import deque
from contextlib import contextmanager


class StageProfiler:
    """Accumulate per-stage timings per frame, then commit/log/publish on flush().

    Typical use inside a node callback::

        with self.profiler.measure("detection"):
            self.detect_objects(image)
        self.profiler.record_speed(self.results)
        ...
        self.profiler.flush(self)

    `measure()` *adds* into the current frame's accumulator, so a stage invoked
    several times in a loop sums into a single per-frame sample. `flush()` commits
    each accumulated stage into a rolling window, optionally publishes the raw
    per-frame values, and optionally emits a throttled aggregate log line.
    """

    def __init__(self, enabled=False, log=True, publish=False,
                 log_interval=5.0, window=200, publisher=None):
        self.enabled = enabled
        self.log = log
        self.publish = publish
        self.log_interval = log_interval
        self.window = window
        self._publisher = publisher

        # rolling history of committed ms per stage
        self._history = {}
        # insertion order of stages (stable column order for the published array)
        self._order = []
        # current frame's accumulator: stage -> ms
        self._frame = {}
        # most-recent committed value per stage
        self._last = {}
        # last aggregate-log time, in seconds (node clock); set on first flush
        self._last_log_t = None

    # ------------------------------------------------------------ measurement

    @contextmanager
    def measure(self, stage):
        """Time the wrapped block and add elapsed ms into the current frame.

        Near-zero overhead when disabled (just yields).
        """
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - t0) * 1000.0)

    def record(self, stage, ms):
        """Add an externally measured value (ms) into the current frame accumulator."""
        if not self.enabled:
            return
        if stage not in self._history:
            self._history[stage] = deque(maxlen=self.window)
            self._order.append(stage)
        self._frame[stage] = self._frame.get(stage, 0.0) + float(ms)

    def record_speed(self, results, prefix="det"):
        """Record Ultralytics `Results.speed` (preprocess/inference/postprocess ms).

        Stored as `<prefix>_pre`, `<prefix>_inf`, `<prefix>_post`. Guards missing
        results/keys so it is safe to call unconditionally.
        """
        if not self.enabled or not results:
            return
        try:
            speed = results[0].speed
        except (IndexError, AttributeError, TypeError):
            return
        if not isinstance(speed, dict):
            return
        for key, suffix in (('preprocess', 'pre'), ('inference', 'inf'),
                            ('postprocess', 'post')):
            val = speed.get(key)
            if val is not None:
                self.record(f'{prefix}_{suffix}', val)

    # ------------------------------------------------------------ commit / output

    def flush(self, node):
        """Commit the current frame's accumulated stages, then publish/log.

        Should be called once at the end of every callback, regardless of whether
        the node published a 3D message.
        """
        if not self.enabled or not self._frame:
            self._frame = {}
            return

        committed = {}
        for stage in self._order:
            if stage in self._frame:
                ms = self._frame[stage]
                self._history[stage].append(ms)
                self._last[stage] = ms
                committed[stage] = ms
        self._frame = {}

        if self.publish and self._publisher is not None and committed:
            self._publisher.publish(build_profile_msg(committed))

        if self.log:
            now = node.get_clock().now().nanoseconds * 1e-9
            if self._last_log_t is None:
                self._last_log_t = now
            elif now - self._last_log_t >= self.log_interval:
                self._last_log_t = now
                node.get_logger().info(self._format_stats())

    def _format_stats(self):
        parts = []
        for stage in self._order:
            hist = self._history.get(stage)
            if not hist:
                continue
            vals = list(hist)
            n = len(vals)
            mean = sum(vals) / n
            parts.append(
                f'{stage}: mean={mean:.2f} min={min(vals):.2f} '
                f'max={max(vals):.2f} n={n}')
        return 'profile [ms] | ' + ' | '.join(parts)

    # ------------------------------------------------------------ live setters

    def set_enabled(self, value):
        self.enabled = bool(value)

    def set_log(self, value):
        self.log = bool(value)

    def set_publish(self, value):
        self.publish = bool(value)

    def set_log_interval(self, value):
        self.log_interval = float(value)


def build_profile_msg(values):
    """Build a Float32MultiArray from {stage: ms}; layout.dim labels = stage names."""
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension
    msg = Float32MultiArray()
    for stage, ms in values.items():
        dim = MultiArrayDimension()
        dim.label = stage
        dim.size = 1
        dim.stride = 1
        msg.layout.dim.append(dim)
        msg.data.append(float(ms))
    return msg


def setup_profiler(node, topic="~/profile"):
    """Declare profiling params on `node`, create the publisher, return a StageProfiler.

    Flat underscore param names (no dot-namespace) avoid clashes with the
    `tracker_2d.*` namespace:
      - `profile`              (bool,  False)  master on/off
      - `profile_log`          (bool,  True)   throttled aggregate log
      - `profile_publish`      (bool,  False)  per-frame Float32MultiArray topic
      - `profile_log_interval` (float, 5.0)    seconds between aggregate logs
      - `profile_window`       (int,   200)    rolling-window sample count

    The Float32MultiArray publisher is always created (an idle publisher is
    negligible) so every flag stays runtime-toggleable.
    """
    from std_msgs.msg import Float32MultiArray

    def _declare(name, default):
        if not node.has_parameter(name):
            node.declare_parameter(name, default)
        return node.get_parameter(name).value

    enabled = _declare('profile', False)
    log = _declare('profile_log', True)
    publish = _declare('profile_publish', False)
    log_interval = _declare('profile_log_interval', 5.0)
    window = _declare('profile_window', 200)

    publisher = node.create_publisher(Float32MultiArray, topic, 10)
    return StageProfiler(
        enabled=enabled, log=log, publish=publish,
        log_interval=log_interval, window=window, publisher=publisher)


def apply_profiler_param(profiler, name, value):
    """Forward a single profile_* parameter change to `profiler`. Returns True if handled."""
    if name == 'profile':
        profiler.set_enabled(value)
    elif name == 'profile_log':
        profiler.set_log(value)
    elif name == 'profile_publish':
        profiler.set_publish(value)
    elif name == 'profile_log_interval':
        profiler.set_log_interval(value)
    else:
        return False
    return True
