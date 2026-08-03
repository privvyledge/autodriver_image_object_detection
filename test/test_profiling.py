"""Tests for utils/profiling.py — StageProfiler accumulation, flush, and param plumbing."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from autodriver_image_object_detection.utils.profiling import (  # noqa: E402
    StageProfiler,
    apply_profiler_param,
    build_profile_msg,
)


class FakeClockTime:
    def __init__(self, seconds):
        self.nanoseconds = int(seconds * 1e9)


class FakeClock:
    def __init__(self):
        self.seconds = 0.0

    def now(self):
        return FakeClockTime(self.seconds)


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, msg):
        self.messages.append(msg)


class FakePublisher:
    def __init__(self):
        self.published = []

    def publish(self, msg):
        self.published.append(msg)


class FakeNode:
    """Enough of an rclpy Node for flush() — a clock and a logger."""

    def __init__(self):
        self.clock = FakeClock()
        self.logger = FakeLogger()

    def get_clock(self):
        return self.clock

    def get_logger(self):
        return self.logger


class FakeUltralyticsResult:
    def __init__(self, speed):
        self.speed = speed


class TestDisabledProfiler(unittest.TestCase):
    """When disabled, nothing is recorded and no output is produced."""

    def setUp(self):
        self.profiler = StageProfiler(enabled=False)
        self.node = FakeNode()

    def test_measure_is_a_noop(self):
        with self.profiler.measure('detection'):
            pass
        self.assertEqual(self.profiler._frame, {})
        self.assertEqual(self.profiler._history, {})

    def test_record_is_a_noop(self):
        self.profiler.record('detection', 12.0)
        self.assertEqual(self.profiler._frame, {})

    def test_flush_logs_nothing(self):
        self.profiler.record('detection', 12.0)
        self.profiler.flush(self.node)
        self.assertEqual(self.node.logger.messages, [])

    def test_record_speed_is_a_noop(self):
        self.profiler.record_speed([FakeUltralyticsResult(
            {'preprocess': 1.0, 'inference': 2.0, 'postprocess': 3.0})])
        self.assertEqual(self.profiler._frame, {})


class TestRecordAndMeasure(unittest.TestCase):

    def setUp(self):
        self.profiler = StageProfiler(enabled=True, log=False)

    def test_record_populates_the_frame_accumulator(self):
        self.profiler.record('detection', 12.0)
        self.assertEqual(self.profiler._frame, {'detection': 12.0})

    def test_repeated_records_sum_within_one_frame(self):
        self.profiler.record('projection', 1.0)
        self.profiler.record('projection', 2.5)
        self.assertAlmostEqual(self.profiler._frame['projection'], 3.5)

    def test_measure_records_a_positive_duration(self):
        with self.profiler.measure('detection'):
            sum(range(10000))
        self.assertIn('detection', self.profiler._frame)
        self.assertGreater(self.profiler._frame['detection'], 0.0)

    def test_measure_records_even_when_the_block_raises(self):
        with self.assertRaises(RuntimeError):
            with self.profiler.measure('detection'):
                raise RuntimeError('boom')
        self.assertIn('detection', self.profiler._frame)

    def test_stage_order_is_insertion_order(self):
        self.profiler.record('b', 1.0)
        self.profiler.record('a', 1.0)
        self.profiler.record('b', 1.0)
        self.assertEqual(self.profiler._order, ['b', 'a'])


class TestRecordSpeed(unittest.TestCase):

    def setUp(self):
        self.profiler = StageProfiler(enabled=True, log=False)

    def test_ultralytics_speed_dict_is_recorded_with_prefix(self):
        self.profiler.record_speed([FakeUltralyticsResult(
            {'preprocess': 1.0, 'inference': 2.0, 'postprocess': 3.0})])
        self.assertEqual(self.profiler._frame,
                         {'det_pre': 1.0, 'det_inf': 2.0, 'det_post': 3.0})

    def test_custom_prefix(self):
        self.profiler.record_speed(
            [FakeUltralyticsResult({'inference': 2.0})], prefix='seg')
        self.assertEqual(self.profiler._frame, {'seg_inf': 2.0})

    def test_missing_keys_are_skipped(self):
        self.profiler.record_speed([FakeUltralyticsResult({'inference': 2.0})])
        self.assertEqual(self.profiler._frame, {'det_inf': 2.0})

    def test_malformed_results_are_ignored(self):
        for bad in (None, [], [object()], [FakeUltralyticsResult('not-a-dict')]):
            with self.subTest(bad=bad):
                self.profiler.record_speed(bad)
        self.assertEqual(self.profiler._frame, {})


class TestFlush(unittest.TestCase):

    def setUp(self):
        self.node = FakeNode()

    def test_flush_commits_the_frame_into_history_and_clears_it(self):
        profiler = StageProfiler(enabled=True, log=False)
        profiler.record('detection', 12.0)
        profiler.flush(self.node)
        self.assertEqual(profiler._frame, {})
        self.assertEqual(list(profiler._history['detection']), [12.0])
        self.assertEqual(profiler._last['detection'], 12.0)

    def test_history_is_bounded_by_window(self):
        profiler = StageProfiler(enabled=True, log=False, window=3)
        for i in range(10):
            profiler.record('detection', float(i))
            profiler.flush(self.node)
        self.assertEqual(list(profiler._history['detection']), [7.0, 8.0, 9.0])

    def test_empty_frame_flush_does_nothing(self):
        profiler = StageProfiler(enabled=True, log=True)
        profiler.flush(self.node)
        self.assertEqual(self.node.logger.messages, [])
        self.assertEqual(profiler._history, {})

    def test_logging_is_throttled_by_log_interval(self):
        profiler = StageProfiler(enabled=True, log=True, log_interval=5.0)
        # First flush only seeds the timer; it must not log.
        profiler.record('detection', 10.0)
        profiler.flush(self.node)
        self.assertEqual(len(self.node.logger.messages), 0)

        self.node.clock.seconds = 1.0
        profiler.record('detection', 10.0)
        profiler.flush(self.node)
        self.assertEqual(len(self.node.logger.messages), 0)

        self.node.clock.seconds = 6.0
        profiler.record('detection', 20.0)
        profiler.flush(self.node)
        self.assertEqual(len(self.node.logger.messages), 1)
        self.assertIn('detection', self.node.logger.messages[0])

    def test_log_line_reports_mean_min_max_and_count(self):
        profiler = StageProfiler(enabled=True, log=False)
        for ms in (10.0, 20.0, 30.0):
            profiler.record('detection', ms)
            profiler.flush(self.node)
        stats = profiler._format_stats()
        self.assertIn('mean=20.00', stats)
        self.assertIn('min=10.00', stats)
        self.assertIn('max=30.00', stats)
        self.assertIn('n=3', stats)

    def test_publish_emits_one_message_per_flush(self):
        publisher = FakePublisher()
        profiler = StageProfiler(enabled=True, log=False, publish=True,
                                 publisher=publisher)
        profiler.record('detection', 12.0)
        profiler.record('projection', 3.0)
        profiler.flush(self.node)
        self.assertEqual(len(publisher.published), 1)
        msg = publisher.published[0]
        self.assertEqual([d.label for d in msg.layout.dim], ['detection', 'projection'])
        self.assertEqual(list(msg.data), [12.0, 3.0])

    def test_publish_disabled_emits_nothing(self):
        publisher = FakePublisher()
        profiler = StageProfiler(enabled=True, log=False, publish=False,
                                 publisher=publisher)
        profiler.record('detection', 12.0)
        profiler.flush(self.node)
        self.assertEqual(publisher.published, [])


class TestBuildProfileMsg(unittest.TestCase):

    def test_one_dim_per_stage_in_insertion_order(self):
        msg = build_profile_msg({'detection': 12.0, 'projection': 3.5})
        self.assertEqual([d.label for d in msg.layout.dim], ['detection', 'projection'])
        self.assertTrue(all(d.size == 1 and d.stride == 1 for d in msg.layout.dim))
        self.assertEqual(list(msg.data), [12.0, 3.5])

    def test_empty_input(self):
        msg = build_profile_msg({})
        self.assertEqual(len(msg.layout.dim), 0)
        self.assertEqual(len(msg.data), 0)


class TestApplyProfilerParam(unittest.TestCase):

    def setUp(self):
        self.profiler = StageProfiler(enabled=False, log=False, publish=False,
                                      log_interval=5.0)

    def test_known_params_are_applied(self):
        cases = [
            ('profile', True, 'enabled', True),
            ('profile_log', True, 'log', True),
            ('profile_publish', True, 'publish', True),
            ('profile_log_interval', 2.0, 'log_interval', 2.0),
        ]
        for name, value, attr, expected in cases:
            with self.subTest(name=name):
                self.assertTrue(apply_profiler_param(self.profiler, name, value))
                self.assertEqual(getattr(self.profiler, attr), expected)

    def test_unknown_param_is_not_handled(self):
        self.assertFalse(apply_profiler_param(self.profiler, 'conf_threshold', 0.5))

    def test_profile_window_is_not_live_reconfigurable(self):
        # window sizes the deques at construction; changing it live is not supported.
        self.assertFalse(apply_profiler_param(self.profiler, 'profile_window', 50))

    def test_values_are_coerced_to_their_declared_types(self):
        apply_profiler_param(self.profiler, 'profile', 1)
        self.assertIs(self.profiler.enabled, True)
        apply_profiler_param(self.profiler, 'profile_log_interval', 3)
        self.assertIsInstance(self.profiler.log_interval, float)


if __name__ == '__main__':
    unittest.main()
