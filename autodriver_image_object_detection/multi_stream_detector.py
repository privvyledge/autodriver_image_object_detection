"""Multi-stream detector: batch YOLO inference across N cameras with a single model instance."""
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from message_filters import Subscriber, TimeSynchronizer, ApproximateTimeSynchronizer
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray

from autodriver_image_object_detection.base_detector import BaseDetector
from autodriver_image_object_detection.utils.imaging_utils import parse_image_message


class MultiStreamDetector(BaseDetector):

    def _param_defaults(self) -> dict:
        return {
            'model_path': 'yolo11n-seg.pt',
            'conf_thresh': 0.25,
            'iou_thresh': 0.45,
            'max_det': 300,
            'classes': ['person', 'car'],
            'track_2d': False,
        }

    def _extra_export_kwargs(self) -> dict:
        return {'batch': self.num_cameras}

    def __init__(self):
        super().__init__('multi_stream_detector')

        # Common params
        self.declare_common_params()

        # Multi-stream-specific params
        self.declare_parameter('num_cameras', 2)
        self.declare_parameter('fps', 30)
        _fps = self.get_parameter('fps').get_parameter_value().integer_value
        self.declare_parameter(
            'synchronization_interval', 1.5 / _fps,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description='<0: no sync (per-camera callbacks + timer); '
                            '0: exact sync; >0: approximate sync with this slop.',
            ))
        self.declare_parameter('input_image_topic_is_compressed', [False, False])
        self.declare_parameter('subscribe_camera_info', False)

        # Read common params
        self._read_common_params()

        # Read multi-stream params
        gp = self.get_parameter
        self.num_cameras = gp('num_cameras').get_parameter_value().integer_value
        self.fps = gp('fps').get_parameter_value().integer_value
        self.synchronization_interval = gp('synchronization_interval').get_parameter_value().double_value
        self.input_image_topic_is_compressed = list(
            gp('input_image_topic_is_compressed').get_parameter_value().bool_array_value)
        self.subscribe_camera_info = gp('subscribe_camera_info').get_parameter_value().bool_value

        # Per-camera state (overrides self.camera_info = None from base)
        self.cameras = tuple(f'stream_{i}' for i in range(self.num_cameras))
        _init = {cam: None for cam in self.cameras}
        self.camera_info = _init.copy()
        self.images = _init.copy()
        self.headers = _init.copy()
        self.msg_metadata = _init.copy()
        self.image_widths = _init.copy()
        self.image_heights = _init.copy()
        self.imgszs = _init.copy()

        # Device + model (_extra_export_kwargs uses self.num_cameras, so must be set first)
        self._setup_device()
        self.load_model()

        qos_profile = self._build_qos_profile()
        sensor_qos = self._build_sensor_qos_profile()

        # Subscribers
        self.subscriptions_ = []
        # Store image_message_format per-camera
        self.image_message_formats = {}
        self.image_message_types = {}

        for i, camera in enumerate(self.cameras):
            compressed = self.input_image_topic_is_compressed[i]
            fmt = 'compressed' if compressed else 'raw'
            msg_type = CompressedImage if compressed else Image
            self.image_message_formats[camera] = fmt
            self.image_message_types[camera] = msg_type

            if self.synchronization_interval >= 0.0:
                img_sub = Subscriber(self, msg_type, f'{camera}/image_raw', qos_profile=sensor_qos,
                                     callback_group=self._sub_cb_group)
                self.subscriptions_.append(img_sub)
                if self.subscribe_camera_info:
                    info_sub = Subscriber(
                        self, CameraInfo, f'{camera}/camera_info', qos_profile=qos_profile,
                        callback_group=self._sub_cb_group)
                    self.subscriptions_.append(info_sub)
            else:
                self.create_subscription(
                    msg_type, f'{camera}/image_raw',
                    lambda msg, idx=i: self.callback_common(msg, idx),
                    qos_profile=sensor_qos,
                    callback_group=self._sub_cb_group)
                self.create_subscription(
                    CameraInfo, f'{camera}/camera_info',
                    lambda msg, cam=camera: self._store_camera_info(msg, cam),
                    qos_profile=qos_profile,
                    callback_group=self._sub_cb_group)

        if self.synchronization_interval >= 0.0:
            if self.synchronization_interval == 0.0:
                self.ts = TimeSynchronizer(self.subscriptions_, self.queue_size)
            else:
                self.ts = ApproximateTimeSynchronizer(
                    self.subscriptions_, self.queue_size, slop=self.synchronization_interval)
            self.ts.registerCallback(self.image_callback_synchronized)
        else:
            self.timer = self.create_timer(0.1, self.timer_callback)

        # Publishers — keyed by camera name for clarity
        self.detection_pubs = {}
        self.debug_pubs = {}
        for camera in self.cameras:
            self.detection_pubs[camera] = self.create_publisher(
                Detection2DArray, f'{camera}/yolo/detection/results', self.queue_size)
            if self.publish_debug_image:
                self.debug_pubs[camera] = {
                    'detection': self.create_publisher(
                        Image, f'{camera}/yolo/detection/debug_image', self.queue_size),
                    'segmentation': self.create_publisher(
                        Image, f'{camera}/yolo/detection/segmentation_image', self.queue_size),
                    'mask': self.create_publisher(
                        Image, f'{camera}/yolo/detection/segmentation_mask', self.queue_size),
                }

        self.get_logger().info(
            f'multi_stream_detector started on {self.device} with {self.num_cameras} camera(s).')

    # ---------------------------------------------------------------- helpers

    def _store_camera_info(self, msg, camera):
        self.camera_info[camera] = msg

    def detect_objects(self, images) -> None:
        """Override to set imgsz from the first available camera before calling base."""
        self.inference_dict['imgsz'] = next(
            (v for v in self.imgszs.values() if v is not None), (640, 640))
        super().detect_objects(images)

    def _publish_debug_images(self, camera, detection_image, mask_img):
        if not self.publish_debug_image or detection_image is None:
            return
        meta = self.msg_metadata[camera]
        if meta is None:
            return
        pubs = self.debug_pubs.get(camera, {})
        header = self.headers[camera]

        det_img = cv2.cvtColor(detection_image, meta['inverse_conversion']) if meta.get('inverse_conversion') else detection_image
        det_msg = self.bridge.cv2_to_imgmsg(det_img, encoding=meta['msg_fmt'])
        det_msg.header = header
        pubs['detection'].publish(det_msg)

        if mask_img is not None:
            mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding='mono8')
            mask_msg.header = header
            pubs['mask'].publish(mask_msg)

            src = self.images[camera]
            src_inv = cv2.cvtColor(src, meta['inverse_conversion']) if meta.get('inverse_conversion') else src
            color_mask = cv2.bitwise_and(src_inv, src_inv, mask=mask_img)
            cmask_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding=meta['msg_fmt'])
            cmask_msg.header = header
            pubs['segmentation'].publish(cmask_msg)

    # --------------------------------------------------------------- callbacks

    def callback_common(self, msg, idx):
        """Parse an image message and store it for the given camera index."""
        camera = self.cameras[idx]
        try:
            fmt = self.image_message_formats[camera]
            (cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt,
             conversion, inverse_conversion, is_color, is_depth,
             compressed_msg_codec) = parse_image_message(
                msg, self.bridge, fmt, logger=self.get_logger())

            if self.imgszs[camera] is None:
                h, w = cv_image.shape[:2]
                self.image_heights[camera] = h
                self.image_widths[camera] = w
                if self.use_image_dimensions:
                    new_h = h if h % 32 == 0 else ((h // 32) + 1) * 32
                    new_w = w if w % 32 == 0 else ((w // 32) + 1) * 32
                    self.imgszs[camera] = (new_h, new_w)
                else:
                    self.imgszs[camera] = (640, 640)

            if (self.resize_image and self.use_image_dimensions
                    and (self.image_heights[camera], self.image_widths[camera])
                    != (self.imgszs[camera][0], self.imgszs[camera][1])):
                cv_image = cv2.resize(
                    cv_image, (self.imgszs[camera][1], self.imgszs[camera][0]),
                    interpolation=cv2.INTER_LINEAR)

            self.images[camera] = cv_image
            self.headers[camera] = msg.header
            self.msg_metadata[camera] = {
                'msg_encoding': msg_encoding,
                'msg_timestamp': msg_timestamp,
                'msg_fmt': msg_fmt,
                'conversion': conversion,
                'inverse_conversion': inverse_conversion,
                'is_color': is_color,
                'is_depth': is_depth,
                'compressed_msg_codec': compressed_msg_codec,
            }
        except Exception as e:
            self.get_logger().error(f'Error processing image for {camera}: {e}')

    def image_callback_synchronized(self, *msgs):
        """Synchronized callback: parse all camera images then run batch inference."""
        step = 2 if self.subscribe_camera_info else 1
        for i, camera in enumerate(self.cameras):
            self.callback_common(msgs[i * step], i)
        self._run_batch_inference()

    def timer_callback(self):
        """Timer-based callback: run batch inference when all cameras have a frame."""
        if any(img is None for img in self.images.values()):
            return
        self._run_batch_inference()

    def _run_batch_inference(self):
        images = list(self.images.values())
        self.detect_objects(images)
        if self.results is None:
            return
        for i, (result, camera) in enumerate(zip(self.results, self.cameras)):
            detections_msg, detection_image, mask_img = super().create_detections_array(
                result, self.headers[camera])
            self.detection_pubs[camera].publish(detections_msg)
            self._publish_debug_images(camera, detection_image, mask_img)

    # ------------------------------------------------------- parameter callback

    def parameter_change_callback(self, params):
        result = super().parameter_change_callback(params)
        for param in params:
            self.get_logger().info(
                f'Param {param.name} → {param.value}: success={result.successful}')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MultiStreamDetector()
    try:
        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(node)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        node.get_logger().info('Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()