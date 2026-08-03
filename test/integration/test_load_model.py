"""`BaseDetector.load_model()` backend-lifetime tests.

An exported model (.engine/.onnx) is validated by constructing it and reading
`.names`, which initialises the inference backend. If the final model is then
constructed separately, two full backends are resident at once — enough to OOM at
startup on a memory-constrained GPU. These tests pin the instance count instead of
relying on a device that is big enough to hide the problem.

No real weights or GPU are needed: the ultralytics model class is swapped for a
counting fake.
"""
import pytest


class FakeModel:
    """Stands in for ultralytics.YOLO. Records every construction and export."""

    instances = []
    exports = []

    def __init__(self, path, names=None, raise_on_init=False):
        if raise_on_init:
            raise RuntimeError('deserialisation failed')
        self.path = path
        self.names = {0: 'person'} if names is None else names
        FakeModel.instances.append(self)

    def to(self, device):
        return self

    def fuse(self):
        pass

    def export(self, **kwargs):
        FakeModel.exports.append((self.path, kwargs))
        return self.path


@pytest.fixture(autouse=True)
def reset_fake():
    FakeModel.instances = []
    FakeModel.exports = []
    yield


@pytest.fixture
def detector(make_node):
    """A bare BaseDetector with common params read and the device set up."""
    from autodriver_image_object_detection.base_detector import BaseDetector

    class _Detector(BaseDetector):
        def __init__(self):
            super().__init__('load_model_test')
            self.declare_common_params()
            self._read_common_params()
            self._setup_device()

    return make_node(_Detector, use_gpu=False, track_2d=False,
                     classes=['person'], model_path='fake_model.engine')


def _install(detector, factory):
    detector._select_model_class = lambda: factory


class TestExportedModelLoad:
    def test_valid_engine_is_loaded_exactly_once(self, detector):
        _install(detector, FakeModel)
        detector.load_model()

        assert len(FakeModel.instances) == 1, (
            'the validated probe must be reused as self.model, not loaded twice')
        assert detector.model is FakeModel.instances[0]
        assert FakeModel.exports == []

    def test_unloadable_engine_triggers_one_export_then_a_fresh_load(self, detector):
        state = {'first': True}

        def factory(path):
            if state['first'] and path.endswith('.engine'):
                state['first'] = False
                raise RuntimeError('deserialisation failed')
            return FakeModel(path)

        _install(detector, factory)
        detector.load_model()

        assert len(FakeModel.exports) == 1
        assert FakeModel.exports[0][0].endswith('.pt')
        # The failed probe is not reused; the final model is a fresh .engine load.
        assert detector.model is FakeModel.instances[-1]
        assert detector.model.path.endswith('.engine')

    def test_engine_with_empty_names_is_re_exported(self, detector):
        def factory(path):
            if path.endswith('.engine') and not FakeModel.instances:
                return FakeModel(path, names={})
            return FakeModel(path)

        _install(detector, factory)
        detector.load_model()

        assert len(FakeModel.exports) == 1
        assert detector.model.names


class TestPlainWeightsLoad:
    def test_pt_path_skips_the_probe_entirely(self, make_node):
        from autodriver_image_object_detection.base_detector import BaseDetector

        class _Detector(BaseDetector):
            def __init__(self):
                super().__init__('load_model_pt_test')
                self.declare_common_params()
                self._read_common_params()
                self._setup_device()

        node = make_node(_Detector, use_gpu=False, track_2d=False,
                         classes=['person'], model_path='fake_model.pt')
        node._select_model_class = lambda: FakeModel
        node.load_model()

        assert len(FakeModel.instances) == 1
        assert FakeModel.exports == []
