"""Node-level construction and dynamic-reconfigure tests.

Grown from the verification probe used for the BaseDetector unification: build the
real node, then assert the post-``__init__`` configuration and the parameter
callback rather than diffing a printed dump by eye.
"""
import pytest


@pytest.fixture
def yolo_node(make_node, model_path):
    from autodriver_image_object_detection.yolo_detection_node import ImageObstacleDetectionNode
    return make_node(
        ImageObstacleDetectionNode,
        model_path=model_path,
        use_gpu=False,
        half_precision=False,
        project_to_3d=True,
        use_depth=True,
        use_pointcloud=True,
        output_frame='sensor_kit_link',
        depth_scale=1000.0,
    )


@pytest.fixture
def single_stream_node(make_node, model_path):
    from autodriver_image_object_detection.single_stream_detector import SingleStreamDetector
    return make_node(
        SingleStreamDetector,
        model_path=model_path,
        use_gpu=False,
        half_precision=False,
    )


@pytest.fixture(params=['yolo_detection_node', 'single_stream_detector'])
def detector_node(request, make_node, model_path):
    """A detector node of each class that carries the shared BaseDetector params."""
    if request.param == 'yolo_detection_node':
        from autodriver_image_object_detection.yolo_detection_node import (
            ImageObstacleDetectionNode as cls,
        )
        extra = {'project_to_3d': False, 'use_depth': False, 'use_pointcloud': False}
    else:
        from autodriver_image_object_detection.single_stream_detector import (
            SingleStreamDetector as cls,
        )
        extra = {}
    return make_node(cls, model_path=model_path, use_gpu=False,
                     half_precision=False, **extra)


class TestYoloNodeConfig:
    def test_device_demotes_to_cpu(self, yolo_node):
        assert yolo_node.device == 'cpu'
        assert yolo_node.use_gpu is False
        assert yolo_node.inference_dict['device'] == 'cpu'

    def test_segmentation_task_inferred_from_model_path(self, yolo_node):
        assert yolo_node.use_segmentation is True
        assert yolo_node.task == 'segment'

    def test_classes_resolved_to_ids(self, yolo_node):
        assert all(isinstance(c, int) for c in yolo_node.classes)
        assert yolo_node.inference_dict['classes'] == yolo_node.classes
        assert yolo_node.class_names[yolo_node.class_names_inv['person']] == 'person'

    def test_inference_dict_tracks_params(self, yolo_node):
        assert yolo_node.inference_dict['conf'] == yolo_node.conf_thresh
        assert yolo_node.inference_dict['iou'] == yolo_node.iou_thresh
        assert yolo_node.inference_dict['max_det'] == yolo_node.max_det
        assert yolo_node.inference_dict['half'] is False

    def test_tracker_config_written_to_temp_file(self, yolo_node):
        if not yolo_node.track_2d:
            pytest.skip('tracking disabled')
        path = yolo_node.tracker_2d_cfg['path']
        assert path.endswith('.yaml')
        assert path != yolo_node.get_parameter('tracker_2d.path').value


class TestParameterCallback:
    def _set(self, node, name, ptype, value):
        from rclpy.parameter import Parameter
        return node.parameter_change_callback([Parameter(name, ptype, value)])

    def test_common_param_reaches_inference_dict(self, yolo_node):
        from rclpy.parameter import Parameter
        result = self._set(yolo_node, 'conf_thresh', Parameter.Type.DOUBLE, 0.7)
        assert result.successful
        assert yolo_node.inference_dict['conf'] == 0.7

    def test_node_specific_param_accepted(self, yolo_node):
        from rclpy.parameter import Parameter
        result = self._set(yolo_node, 'cluster_selection', Parameter.Type.STRING, 'closest')
        assert result.successful
        assert yolo_node.cluster_selection == 'closest'

    def test_node_specific_param_validated(self, yolo_node):
        from rclpy.parameter import Parameter
        result = self._set(yolo_node, 'cluster_selection', Parameter.Type.STRING, 'bogus')
        assert not result.successful
        assert yolo_node.cluster_selection != 'bogus'

    def test_non_reconfigurable_param_rejected(self, yolo_node):
        from rclpy.parameter import Parameter
        result = self._set(yolo_node, 'output_frame', Parameter.Type.STRING, 'x')
        assert not result.successful


class TestUpdateClass:
    """`update_class` lives in BaseDetector — exercise it through both subclasses.

    A subclass that rejects params it does not recognise (yolo_detection_node) will
    fail these unless `update_class` is in COMMON_RECONFIGURABLE_PARAMS.
    """

    def _update(self, node, value):
        from rclpy.parameter import Parameter
        return node.parameter_change_callback(
            [Parameter('update_class', Parameter.Type.STRING, value)])

    def test_add_class(self, detector_node):
        target = 'chair'
        key = detector_node.class_names_inv[target]
        if key in detector_node.classes:
            detector_node.classes.remove(key)
        result = self._update(detector_node, target)
        assert result.successful
        assert key in detector_node.classes
        assert detector_node.inference_dict['classes'] == detector_node.classes

    def test_remove_class(self, detector_node):
        key = detector_node.class_names_inv['person']
        if key not in detector_node.classes:
            detector_node.classes.append(key)
        result = self._update(detector_node, '-person')
        assert result.successful
        assert key not in detector_node.classes
        assert detector_node.inference_dict['classes'] == detector_node.classes

    def test_unknown_class_rejected_without_corrupting_list(self, detector_node):
        before = list(detector_node.classes)
        result = self._update(detector_node, 'definitely_not_a_class')
        assert not result.successful
        assert detector_node.classes == before

    def test_classes_param_kept_in_sync(self, detector_node):
        self._update(detector_node, 'chair')
        names = list(detector_node.get_parameter('classes').value)
        assert sorted(names) == sorted(detector_node.class_names[c]
                                       for c in detector_node.classes)
