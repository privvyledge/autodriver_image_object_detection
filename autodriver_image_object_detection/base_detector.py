"""Base class for YOLO-based ROS2 detection nodes."""
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
import yaml

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from vision_msgs.msg import Detection2DArray

from autodriver_image_object_detection.utils.common import (
    pack_2d_detection,
    update_tracker_param,
)


class BaseDetector(Node):
    """Shared infrastructure for YOLO-based ROS2 detection nodes.

    Subclass usage::

        class MyDetector(BaseDetector):
            def __init__(self):
                super().__init__('my_detector')
                self.declare_common_params()
                # declare node-specific params here
                self._read_common_params()
                # read and act on node-specific params here
                self._setup_device()
                self.load_model()
                # set up subscribers / publishers here
                self.add_on_set_parameters_callback(self.parameter_change_callback)

    To override default parameter values without touching declare_common_params,
    override _param_defaults() and return a dict of {param_name: default_value}.
    """

    def __init__(self, node_name: str):
        super().__init__(node_name)
        self._init_callback_groups()

    def _init_callback_groups(self) -> None:
        self._sub_cb_group = MutuallyExclusiveCallbackGroup()
        self._param_cb_group = MutuallyExclusiveCallbackGroup()

    # ---------------------------------------------------------- param defaults

    def _param_defaults(self) -> dict:
        """Return default parameter values. Override in subclasses for node-specific defaults."""
        return {}

    # ---------------------------------------------------- parameter declaration

    def declare_common_params(self) -> None:
        """Declare all parameters shared across detector nodes."""
        pkg = get_package_share_directory('autodriver_image_object_detection')
        d = self._param_defaults()

        self.declare_parameter('qos', d.get('qos', 'SENSOR_DATA'))
        self.declare_parameter('queue_size', d.get('queue_size', 1))
        self.declare_parameter('use_gpu', d.get('use_gpu', True))
        self.declare_parameter('show_image', d.get('show_image', False))
        self.declare_parameter('publish_debug_image', d.get('publish_debug_image', True))
        self.declare_parameter(
            'model_path',
            d.get('model_path', 'yolo11n-seg.engine'),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description='Path to an Ultralytics model file (.pt, .engine, .onnx). '
                            'If a .engine/.onnx is missing the node auto-exports from the .pt.',
            ),
        )
        self.declare_parameter('export_model_format', d.get('export_model_format', ''))
        self.declare_parameter(
            'export_and_exit',
            d.get('export_and_exit', False),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='If True, shut the node down right after a model export/auto-export '
                            'completes instead of continuing to run inference. Useful for a '
                            'dedicated "build the .engine" pass.',
            ),
        )
        self.declare_parameter('half_precision', d.get('half_precision', True))
        self.declare_parameter('conf_thresh', d.get('conf_thresh', 0.55))
        self.declare_parameter('iou_thresh', d.get('iou_thresh', 0.55))
        self.declare_parameter('max_det', d.get('max_det', 50))
        self.declare_parameter(
            'classes',
            d.get('classes', ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck']),
        )
        self.declare_parameter('agnostic_nms', d.get('agnostic_nms', True))
        self.declare_parameter('augment', d.get('augment', False))
        self.declare_parameter('verbose', d.get('verbose', False))
        self.declare_parameter(
            'use_image_dimensions',
            d.get('use_image_dimensions', True),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_BOOL,
                description='Pass actual image H×W (rounded up to 32-multiple) to YOLO. '
                            'Set False to use fixed image_dimensions instead.',
            ),
        )
        self.declare_parameter(
            'image_dimensions',
            d.get('image_dimensions', [640, 640]),
            ParameterDescriptor(
                type=ParameterType.PARAMETER_INTEGER_ARRAY,
                description='[H, W] used when exporting a model or when use_image_dimensions=False.',
            ),
        )
        self.declare_parameter('resize_image', d.get('resize_image', False))
        self.declare_parameter('track_2d', d.get('track_2d', True))
        self.declare_parameter(
            'tracker_2d.path',
            d.get('tracker_2d.path', os.path.join(pkg, 'config', 'tracker_orin_nano.yaml')),
        )
        self.declare_parameter('tracker_2d.tracker_type', d.get('tracker_2d.tracker_type', 'bytetrack'))
        self.declare_parameter('tracker_2d.track_high_thresh', d.get('tracker_2d.track_high_thresh', -1.0))
        self.declare_parameter('tracker_2d.track_low_thresh', d.get('tracker_2d.track_low_thresh', -1.0))
        self.declare_parameter('tracker_2d.new_track_thresh', d.get('tracker_2d.new_track_thresh', -1.0))
        self.declare_parameter('tracker_2d.track_buffer', d.get('tracker_2d.track_buffer', -1))
        self.declare_parameter('tracker_2d.match_thresh', d.get('tracker_2d.match_thresh', -1.0))
        self.declare_parameter('tracker_2d.fuse_score', d.get('tracker_2d.fuse_score', True))
        self.declare_parameter('tracker_2d.gmc_method', d.get('tracker_2d.gmc_method', ''))
        self.declare_parameter('tracker_2d.proximity_thresh', d.get('tracker_2d.proximity_thresh', -1.0))
        self.declare_parameter('tracker_2d.appearance_thresh', d.get('tracker_2d.appearance_thresh', -1.0))
        self.declare_parameter('tracker_2d.with_reid', d.get('tracker_2d.with_reid', True))
        self.declare_parameter('tracker_2d.model', d.get('tracker_2d.model', 'auto'))
        self.declare_parameter('plot_tracks', d.get('plot_tracks', True))
        self.declare_parameter('static_camera_info', d.get('static_camera_info', True))

    def _read_common_params(self) -> None:
        """Read all common parameters into self.* and initialise shared state."""
        gp = self.get_parameter
        self.qos = gp('qos').get_parameter_value().string_value
        self.queue_size = gp('queue_size').get_parameter_value().integer_value
        self.use_gpu = gp('use_gpu').get_parameter_value().bool_value
        self.show_image = gp('show_image').get_parameter_value().bool_value
        self.publish_debug_image = gp('publish_debug_image').get_parameter_value().bool_value
        self.model_path = gp('model_path').get_parameter_value().string_value
        self.export_model_format = gp('export_model_format').get_parameter_value().string_value
        self.export_and_exit = gp('export_and_exit').get_parameter_value().bool_value
        self.half_precision = gp('half_precision').get_parameter_value().bool_value
        self.conf_thresh = gp('conf_thresh').get_parameter_value().double_value
        self.iou_thresh = gp('iou_thresh').get_parameter_value().double_value
        self.max_det = gp('max_det').get_parameter_value().integer_value
        self.classes = list(gp('classes').value)
        self.agnostic_nms = gp('agnostic_nms').get_parameter_value().bool_value
        self.augment = gp('augment').get_parameter_value().bool_value
        self.verbose = gp('verbose').get_parameter_value().bool_value
        self.use_image_dimensions = gp('use_image_dimensions').get_parameter_value().bool_value
        self.image_dimensions = list(gp('image_dimensions').get_parameter_value().integer_array_value)
        self.resize_image = gp('resize_image').get_parameter_value().bool_value
        self.track_2d = gp('track_2d').get_parameter_value().bool_value
        self.tracker_2d_cfg = {k: v.value for k, v in self.get_parameters_by_prefix('tracker_2d').items()}
        self.plot_tracks = gp('plot_tracks').get_parameter_value().bool_value
        self.static_camera_info = gp('static_camera_info').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.results = None
        self.camera_info = None

    # ----------------------------------------------------------- device setup

    def _setup_device(self) -> None:
        """Initialise self.device and self.torch_device; demote use_gpu if CUDA absent."""
        self.device = 'cpu'
        self.torch_device = torch.device('cpu')
        if self.use_gpu:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
                self.torch_device = torch.device('cuda:0')
            else:
                self.use_gpu = False

    # --------------------------------------------------------------- model

    def _select_model_class(self):
        """Return the ultralytics model class matching self.model_path."""
        mp = self.model_path.lower()
        if 'rtdetr' in mp:
            return ultralytics.RTDETR
        if 'nas' in mp:
            return ultralytics.NAS
        if 'worldv2' in mp:
            return ultralytics.YOLOWorld
        return ultralytics.YOLO

    def _extra_export_kwargs(self) -> dict:
        """Extra kwargs forwarded to model.export(). Override for batch=N in multi-stream."""
        return {}

    def load_model(self) -> None:
        """Load the YOLO model, resolve classes, build inference dict, and configure tracker."""
        os.environ['YOLO_VERBOSE'] = str(self.verbose)
        self.use_segmentation = 'seg' in self.model_path
        self.task = 'segment' if self.use_segmentation else 'detect'

        model_class = self._select_model_class()
        model_path = Path(self.model_path)
        extra = self._extra_export_kwargs()
        did_export = False

        if self.export_model_format:
            self.get_logger().info(f'Exporting model to {self.export_model_format}...')
            model_class(str(model_path.with_suffix('.pt'))).export(
                format=self.export_model_format,
                half=self.half_precision,
                simplify=True,
                nms=self.iou_thresh > 0.0,
                dynamic=True,
                device=self.device,
                **extra,
            )
            self.model_path = str(model_path.with_suffix('.' + self.export_model_format))
            model_path = Path(self.model_path)
            did_export = True
            self.get_logger().info(f'Export complete: {self.model_path}')

        if model_path.suffix in ('.engine', '.onnx'):
            needs_export = False
            # Do NOT pre-check model_path.exists(): a relative model_path resolves against
            # the launch cwd, so .exists() returns False for an engine Ultralytics could
            # still find via its own asset/search dirs — forcing a needless ~5-min
            # re-export on every startup. Let the probe decide instead.
            try:
                probe_model = model_class(str(model_path))
                # Trigger the actual deserialisation/load by accessing names.
                # NOTE: do NOT treat `probe_model.model` being a str path as a failure.
                # Ultralytics keeps the weights path as a string for exported formats
                # (.engine/.onnx) until the inference backend is lazily initialised, so
                # that check fired on EVERY valid engine and forced a needless re-export.
                names = probe_model.names
                if not names:
                    needs_export = True
            except Exception as e:
                self.get_logger().warn(f'Failed to load {model_path}: {e}. Will attempt to re-export.')
                needs_export = True

            if needs_export:
                fmt = model_path.suffix.lstrip('.')
                self.get_logger().info(f'{model_path} failed to load or does not exist. Exporting from .pt...')
                model_class(str(model_path.with_suffix('.pt'))).export(
                    format=fmt,
                    half=self.half_precision,
                    simplify=True,
                    nms=self.iou_thresh > 0.0,
                    dynamic=True,
                    device=self.device,
                    **extra,
                )
                did_export = True

        if did_export and self.export_and_exit:
            self.get_logger().info('export_and_exit=True — export finished; shutting node down.')
            raise SystemExit(0)

        self.model = model_class(str(model_path))
        try:
            self.model.to(self.torch_device)
        except TypeError:
            pass

        self.class_names: dict[int, str] = self.model.names
        self.class_names_inv: dict[str, int] = {v: k for k, v in self.class_names.items()}
        self.supported_class_names = set(self.class_names_inv)
        self.supported_class_keys = set(self.class_names)
        self._resolve_classes()
        self.get_logger().info(f'Detecting classes: {[self.class_names[c] for c in self.classes]}')

        try:
            self.get_logger().info('Fusing model...')
            self.model.fuse()
        except TypeError as e:
            self.get_logger().warn(f'Model fuse skipped (normal for non-.pt): {e}')

        self._setup_inference_dict()

        # A fused .pt model moved to GPU keeps fp32 weights, while half=True feeds fp16
        # input — raising "mat1 and mat2 have different dtype: Half != float". Match the
        # weight dtype to the requested precision for the PyTorch path (engines/onnx are
        # already exported at the right precision and need no change).
        if (self.use_gpu and self.half_precision
                and Path(self.model_path).suffix == '.pt'):
            try:
                self.model.model.half()
            except Exception as e:
                self.get_logger().warn(
                    f'Could not cast .pt model to fp16 ({e}); falling back to fp32 inference.')
                self.inference_dict['half'] = False

        if self.track_2d and self.plot_tracks:
            self.track_history: dict = defaultdict(list)

        if self.track_2d:
            self._setup_tracker_config()

    def _resolve_classes(self) -> None:
        """Normalise self.classes to a list of integer class IDs."""
        classes = self.classes
        if not classes:
            self.classes = list(range(len(self.class_names)))
            return
        if isinstance(classes, int):
            self.classes = [classes]
            return
        if isinstance(classes, str):
            self.classes = [int(x.strip()) for x in classes.split(',')]
            return
        if isinstance(classes, (list, tuple)):
            classes = [c for c in classes if c or c == 0]
            if classes and isinstance(classes[0], str):
                assert all(c in self.supported_class_names for c in classes)
                self.classes = [self.class_names_inv[c.strip()] for c in classes]
            else:
                assert all(c in self.supported_class_keys for c in classes)
                self.classes = [int(c) for c in classes]
            return
        self.classes = list(classes)

    def _setup_inference_dict(self) -> None:
        """Initialise self.imgsz and self.inference_dict for detect_objects()."""
        self.imgsz = None
        self.inference_dict = {
            'source': None,
            'conf': self.conf_thresh,
            'iou': self.iou_thresh,
            'imgsz': self.imgsz,
            'device': self.device,
            'half': self.half_precision,
            'classes': self.classes,
            'max_det': self.max_det,
            'retina_masks': True,
            'show': False,
            'stream': False,
            'augment': self.augment,
            'agnostic_nms': self.agnostic_nms,
            'verbose': self.verbose,
        }

    def _setup_tracker_config(self) -> None:
        """Merge tracker_2d.* params over the base YAML and write to a temp file."""
        with open(self.tracker_2d_cfg['path'], 'r') as f:
            tracker_cfg = yaml.safe_load(f)
        for k, new_v in self.tracker_2d_cfg.items():
            if k in tracker_cfg:
                tracker_cfg[k] = update_tracker_param(k, new_value=new_v, old_value=tracker_cfg[k])
        assert tracker_cfg['tracker_type'] in ('bytetrack', 'botsort'), \
            f"Unsupported tracker type: {tracker_cfg['tracker_type']}"
        self._tracker_temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml')
        yaml.safe_dump(tracker_cfg, self._tracker_temp_file, default_flow_style=False, sort_keys=False)
        # Flush before Ultralytics reads the path, or it may see stale/empty content.
        self._tracker_temp_file.flush()
        self.tracker_2d_cfg['path'] = self._tracker_temp_file.name

    # -------------------------------------------------------------- QoS

    def _build_qos_profile(self) -> QoSProfile:
        """Return a QoSProfile based on self.qos and self.queue_size."""
        if self.qos.lower() == 'sensor_data':
            return QoSProfile(
                reliability=QoSReliabilityPolicy.BEST_EFFORT,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=self.queue_size,
            )
        return QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=self.queue_size,
        )

    def _build_sensor_qos_profile(self) -> QoSProfile:
        """Return BEST_EFFORT/KEEP_LAST/depth=1 — suitable for raw sensor topics."""
        return QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

    # ------------------------------------------------------------ inference

    def detect_objects(self, images) -> None:
        """Run model.track or model.predict on images (ndarray or list of ndarrays).

        Sets self.results to the Ultralytics Results list, or None on error.
        Callers must set self.inference_dict['imgsz'] before calling.
        """
        try:
            self.inference_dict['source'] = images
            if self.track_2d:
                self.results = self.model.track(
                    tracker=self.tracker_2d_cfg['path'],
                    persist=True,
                    **self.inference_dict,
                )
            else:
                self.results = self.model.predict(**self.inference_dict)
        except Exception as e:
            self.get_logger().error(f'Inference error: {e}')
            self.results = None

    def create_detections_array(self, result, header):
        """Parse a single Ultralytics result into a Detection2DArray.

        Renders the annotated image and composites the segmentation mask.
        Appends track history polylines when track_2d and plot_tracks are both True.

        Args:
            result: A single element from the Ultralytics Results list.
            header: The ROS2 Header to stamp the Detection2DArray.

        Returns:
            (Detection2DArray, detection_image ndarray, mask_img ndarray or None)
        """
        detections_msg = Detection2DArray()
        detections_msg.header.stamp = header.stamp
        detections_msg.header.frame_id = header.frame_id

        # result.plot() rasterises boxes/masks/labels onto a fresh image and is
        # expensive (notably on Jetson). Only render when a consumer exists.
        render = self.show_image or self.publish_debug_image
        detection_image = (
            result.plot(conf=True, labels=True, boxes=True, masks=True, probs=True)
            if render else None
        )
        if self.show_image:
            try:
                cv2.imshow('detection', detection_image)
                cv2.waitKey(1)
            except Exception as e:
                self.get_logger().warning(
                    f"Could not display window 'detection' (likely headless environment): {e}. Disabling show_image."
                )
                self.show_image = False

        mask_img = None
        boxes = result.boxes.cpu()

        if boxes.shape[0] < 1:
            return detections_msg, detection_image, mask_img

        if result.masks is not None and render:
            mask_img = (torch.sum(result.masks.data, dim=0).cpu().numpy() * 255).astype(np.uint8)
            if self.show_image:
                try:
                    cv2.imshow('mask', mask_img)
                    cv2.waitKey(1)
                except Exception as e:
                    self.get_logger().warning(
                        f"Could not display window 'mask' (likely headless environment): {e}. Disabling show_image."
                    )
                    self.show_image = False

        track_ids = None
        if self.track_2d and boxes.is_track and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()

        all_xywh = boxes.xywh.numpy()
        all_cls = boxes.cls.numpy().astype(int)
        all_conf = boxes.conf.numpy()

        for i, (bbox, cls, conf) in enumerate(zip(all_xywh, all_cls, all_conf)):
            x, y, w, h = bbox
            track_id = track_ids[i] if track_ids is not None else -1

            if (self.track_2d and self.plot_tracks and detection_image is not None
                    and hasattr(self, 'track_history')):
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                pts = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(detection_image, [pts], isClosed=False, color=(230, 230, 230), thickness=5)

            detections_msg.detections.append(
                pack_2d_detection(x, y, w, h, result.names.get(int(cls)), conf, id=track_id)
            )

        return detections_msg, detection_image, mask_img

    # ------------------------------------------------ parameter change callback

    def parameter_change_callback(self, params):
        """Handle changes to all common parameters.

        Subclasses that add their own parameters should override this method,
        call super().parameter_change_callback(params) first, then handle the
        remaining params themselves.  The base class does NOT mark unrecognised
        parameters as failed so subclass-specific params pass through cleanly.
        """
        result = SetParametersResult(successful=True)
        for param in params:
            name, val, ptype = param.name, param.value, param.type_
            if name == 'publish_debug_image' and ptype == Parameter.Type.BOOL:
                self.publish_debug_image = val
            elif name == 'track_2d' and ptype == Parameter.Type.BOOL:
                self.track_2d = val
            elif name == 'tracker_2d.path' and ptype == Parameter.Type.STRING:
                self.tracker_2d_cfg['path'] = val
            elif name == 'plot_tracks' and ptype == Parameter.Type.BOOL:
                self.plot_tracks = val
            elif name == 'use_gpu' and ptype == Parameter.Type.BOOL:
                self.device = 'cpu'
                self.torch_device = torch.device('cpu')
                self.use_gpu = False
                if val:
                    if torch.cuda.is_available():
                        self.device = 'cuda:0'
                        self.torch_device = torch.device('cuda:0')
                        self.use_gpu = True
                    else:
                        result.successful = False
                        result.reason = 'CUDA not available'
                        self.get_logger().warn('CUDA not available; staying on CPU.')
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['device'] = self.device
            elif name == 'show_image' and ptype == Parameter.Type.BOOL:
                self.show_image = val
                if not val:
                    try:
                        cv2.destroyAllWindows()
                    except Exception:
                        pass
            elif name == 'use_image_dimensions' and ptype == Parameter.Type.BOOL:
                self.use_image_dimensions = val
            elif name == 'image_dimensions' and ptype == Parameter.Type.INTEGER_ARRAY:
                self.image_dimensions = list(val)
            elif name == 'resize_image' and ptype == Parameter.Type.BOOL:
                self.resize_image = val
            elif name == 'half_precision' and ptype == Parameter.Type.BOOL:
                self.half_precision = val
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['half'] = val
            elif name == 'conf_thresh' and ptype == Parameter.Type.DOUBLE:
                self.conf_thresh = val
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['conf'] = val
            elif name == 'iou_thresh' and ptype == Parameter.Type.DOUBLE:
                self.iou_thresh = val
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['iou'] = val
            elif name == 'max_det' and ptype == Parameter.Type.INTEGER:
                self.max_det = val
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['max_det'] = val
            elif name == 'classes' and ptype in (Parameter.Type.STRING_ARRAY, Parameter.Type.INTEGER_ARRAY):
                if ptype == Parameter.Type.STRING_ARRAY:
                    unknown = [c for c in val if c not in self.supported_class_names]
                    if unknown:
                        result.successful = False
                        result.reason = f'Unknown class name(s): {unknown}'
                    else:
                        self.classes = [self.class_names_inv[c.strip()] for c in val]
                        if hasattr(self, 'inference_dict'):
                            self.inference_dict['classes'] = self.classes
                else:
                    unknown = [c for c in val if c not in self.supported_class_keys]
                    if unknown:
                        result.successful = False
                        result.reason = f'Unknown class id(s): {unknown}'
                    else:
                        self.classes = list(val)
                        if hasattr(self, 'inference_dict'):
                            self.inference_dict['classes'] = self.classes
            elif name == 'agnostic_nms' and ptype == Parameter.Type.BOOL:
                self.agnostic_nms = val
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['agnostic_nms'] = val
            elif name == 'augment' and ptype == Parameter.Type.BOOL:
                self.augment = val
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['augment'] = val
            elif name == 'verbose' and ptype == Parameter.Type.BOOL:
                self.verbose = val
                os.environ['YOLO_VERBOSE'] = str(val)
                if hasattr(self, 'inference_dict'):
                    self.inference_dict['verbose'] = val
            elif name == 'static_camera_info' and ptype == Parameter.Type.BOOL:
                self.static_camera_info = val
            self.get_logger().info(f'Param {param.name} → {param.value}: success={result.successful}')
        return result

    # --------------------------------------------------------------- cleanup

    def destroy_node(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if hasattr(self, 'model'):
            del self.model
        if 'cuda' in getattr(self, 'device', ''):
            self.get_logger().info('Clearing CUDA cache')
            torch.cuda.empty_cache()
        if getattr(self, 'track_2d', False) and hasattr(self, '_tracker_temp_file'):
            try:
                self._tracker_temp_file.close()
            except Exception:
                pass
        super().destroy_node()