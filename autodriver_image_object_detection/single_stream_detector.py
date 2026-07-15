"""Single stream detector for object detection with optional tracking."""
import os
import queue
import threading

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.parameter import Parameter
from rcl_interfaces.msg import ParameterDescriptor, ParameterType, SetParametersResult
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray

from autodriver_image_object_detection.base_detector import BaseDetector
from autodriver_image_object_detection.utils.imaging_utils import parse_image_message
from autodriver_image_object_detection.utils.profiling import setup_profiler, apply_profiler_param


class SingleStreamDetector(BaseDetector):
    def __init__(self):
        super().__init__('single_stream_detector')

        # Common params (model, inference, tracking, QoS, etc.)
        self.declare_common_params()

        # Node-specific params
        self.declare_parameter(
            'input_image_topic', 'carla/ego_vehicle/rgb_front/image',
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING,
                                description='Input image topic. Supports all encodings.'))
        self.declare_parameter('input_camera_info_topic', 'carla/ego_vehicle/rgb_front/camera_info')
        self.declare_parameter('input_image_topic_is_compressed', False)
        self.declare_parameter('detection_results_topic', 'yolo/detection_results')
        self.declare_parameter('detection_image_topic', 'yolo/detection_image')
        self.declare_parameter('segmentation_image_topic', 'yolo/segmentation_image')
        self.declare_parameter('segmentation_mask_image_topic', 'yolo/segmentation_mask_image')
        self.declare_parameter('update_class', '')  # "class_name" to add, "-class_name" to remove

        # Read common params into self.*
        self._read_common_params()

        # Read node-specific params
        gp = self.get_parameter
        self.input_image_topic = gp('input_image_topic').get_parameter_value().string_value
        self.input_camera_info_topic = gp('input_camera_info_topic').get_parameter_value().string_value
        self.input_image_topic_is_compressed = gp('input_image_topic_is_compressed').get_parameter_value().bool_value
        self.detection_results_topic = gp('detection_results_topic').get_parameter_value().string_value
        self.detection_image_topic = gp('detection_image_topic').get_parameter_value().string_value
        self.segmentation_image_topic = gp('segmentation_image_topic').get_parameter_value().string_value
        self.segmentation_mask_image_topic = gp('segmentation_mask_image_topic').get_parameter_value().string_value
        self.update_class = gp('update_class').get_parameter_value().string_value

        # Device + model
        self._setup_device()
        self.load_model()
        self.image_frame_id = None
        self.image_width = None
        self.image_height = None

        # Append /compressed suffixes if needed
        if self.input_image_topic_is_compressed:
            for attr in ('input_image_topic', 'detection_image_topic',
                         'segmentation_image_topic', 'segmentation_mask_image_topic'):
                topic = getattr(self, attr)
                if not topic.endswith('/compressed'):
                    setattr(self, attr, topic + '/compressed')

        # Image message format
        self.image_message_format = 'raw'
        self.image_message_type = Image
        if self.input_image_topic_is_compressed or 'compressed' in self.input_image_topic:
            self.image_message_format = 'compressed'
            self.image_message_type = CompressedImage

        qos_profile = self._build_qos_profile()
        sensor_qos = self._build_sensor_qos_profile()

        # Subscribers
        self.image_sub = self.create_subscription(
            self.image_message_type, self.input_image_topic, self.image_callback, sensor_qos,
            callback_group=self._sub_cb_group)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, self.input_camera_info_topic, self.camera_info_callback, qos_profile,
            callback_group=self._sub_cb_group)

        # Publishers
        self.detection_results_pub = self.create_publisher(
            Detection2DArray, self.detection_results_topic, self.queue_size)

        if self.publish_debug_image:
            self.detection_image_pub = self.create_publisher(
                self.image_message_type, self.detection_image_topic, self.queue_size)
            self.segmentation_image_pub = self.create_publisher(
                self.image_message_type, self.segmentation_image_topic, self.queue_size)
            self.segmentation_mask_image_pub = self.create_publisher(
                self.image_message_type, self.segmentation_mask_image_topic, self.queue_size)

        self.profiler = setup_profiler(self)

        self.add_on_set_parameters_callback(self.parameter_change_callback)

        self._image_queue = queue.Queue(maxsize=2)
        self._inference_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self._inference_thread.start()

        self.get_logger().info(
            f'single_stream_detector started. '
            f'Subscribing to {self.input_image_topic}. '
            f'Publishing on {self.detection_results_topic}.'
        )

    # ---------------------------------------------------------------- callbacks

    def image_callback(self, msg):
        if self._image_queue.full():
            try:
                self._image_queue.get_nowait()
            except queue.Empty:
                pass
        try:
            self._image_queue.put_nowait(msg)
        except queue.Full:
            pass

    def _inference_worker(self):
        while rclpy.ok():
            try:
                msg = self._image_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if self.camera_info is None:
                self.get_logger().warn('No CameraInfo received yet — skipping frame.', once=True)
                continue
            try:
                (cv_image, msg_encoding, image_frame_id, msg_timestamp, msg_fmt,
                 conversion, inverse_conversion, is_color, is_depth,
                 compressed_msg_codec) = parse_image_message(
                    msg, self.bridge, self.image_message_format, logger=self.get_logger())

                if self.imgsz is None:
                    self.image_height, self.image_width = cv_image.shape[:2]
                    if self.use_image_dimensions:
                        new_h = (self.image_height if self.image_height % 32 == 0
                                 else ((self.image_height // 32) + 1) * 32)
                        new_w = (self.image_width if self.image_width % 32 == 0
                                 else ((self.image_width // 32) + 1) * 32)
                        self.imgsz = (new_h, new_w)
                        self.get_logger().info(f'Using image dimensions: {self.imgsz}')
                    else:
                        self.imgsz = list(self.image_dimensions)
                self.inference_dict['imgsz'] = self.imgsz

                if (self.resize_image
                        and (self.image_height, self.image_width) != (self.imgsz[0], self.imgsz[1])):
                    cv_image = cv2.resize(cv_image, (self.imgsz[1], self.imgsz[0]))

                with self.profiler.measure("detection"):
                    self.detect_objects(cv_image)
                self.profiler.record_speed(self.results)
                detection_msg, detection_image, mask_img = self.parse_results(self.results, msg.header)

                if detection_msg is None:
                    self.profiler.flush(self)
                    continue
                self.detection_results_pub.publish(detection_msg)

                if detection_image is not None and self.publish_debug_image:
                    if conversion is not None:
                        detection_image = cv2.cvtColor(detection_image, inverse_conversion)

                    if self.image_message_format in ('compressed', 'packet'):
                        det_img_msg = self.bridge.cv2_to_compressed_imgmsg(
                            detection_image, dst_format=compressed_msg_codec)
                    else:
                        det_img_msg = self.bridge.cv2_to_imgmsg(detection_image, encoding=msg_fmt)
                    det_img_msg.header.frame_id = image_frame_id
                    det_img_msg.header.stamp = msg_timestamp

                    if self.detection_image_topic:
                        self.detection_image_pub.publish(det_img_msg)

                    if self.segmentation_mask_image_topic and mask_img is not None:
                        if self.image_message_format in ('compressed', 'packet'):
                            mask_msg = self.bridge.cv2_to_compressed_imgmsg(
                                mask_img, dst_format=compressed_msg_codec)
                        else:
                            mask_msg = self.bridge.cv2_to_imgmsg(mask_img, encoding='mono8')
                        mask_msg.header.frame_id = image_frame_id
                        mask_msg.header.stamp = msg_timestamp
                        self.segmentation_mask_image_pub.publish(mask_msg)

                    if self.segmentation_image_topic and mask_img is not None:
                        cv_image_inv = cv_image
                        if inverse_conversion is not None:
                            cv_image_inv = cv2.cvtColor(cv_image, inverse_conversion)
                        color_mask = cv2.bitwise_and(cv_image_inv, cv_image_inv, mask=mask_img)
                        if self.show_image:
                            try:
                                cv2.imshow('color_mask_image', color_mask)
                                cv2.waitKey(1)
                            except Exception as e:
                                self.get_logger().warning(
                                    f"Could not display window 'color_mask_image' (likely headless environment): {e}. Disabling show_image."
                                )
                                self.show_image = False

                        if self.image_message_format in ('compressed', 'packet'):
                            cmask_msg = self.bridge.cv2_to_compressed_imgmsg(
                                color_mask, dst_format=compressed_msg_codec)
                        else:
                            cmask_msg = self.bridge.cv2_to_imgmsg(color_mask, encoding=msg_fmt)
                        cmask_msg.header.frame_id = image_frame_id
                        cmask_msg.header.stamp = msg_timestamp
                        self.segmentation_image_pub.publish(cmask_msg)

                self.profiler.flush(self)

            except Exception as e:
                self.get_logger().error(f'Error processing image: {e}')

    def camera_info_callback(self, msg):
        if self.camera_info is None or not self.static_camera_info:
            self.camera_info = msg

    # --------------------------------------------------------------- parsing

    def parse_results(self, results, header):
        # Image-only node: detections stay in pixel space and are published as
        # vision_msgs/Detection2DArray. Metric ObjectArray/ObstacleArray output
        # requires a depth or pointcloud projection and lives in yolo_detection_node.
        if results is None:
            return None, None, None
        if not results:
            return Detection2DArray(), None, None
        return self.create_detections_array(results[0], header)

    # ------------------------------------------------------- parameter callback

    def parameter_change_callback(self, params):
        """Handle common params via super, then handle single_stream-specific params."""
        result = super().parameter_change_callback(params)
        for param in params:
            name, val, ptype = param.name, param.value, param.type_
            if apply_profiler_param(self.profiler, name, val):
                pass
            elif name == 'model_path' and ptype == Parameter.Type.STRING:
                self.model_path = val
                # todo: reload model
            elif name == 'update_class' and ptype == Parameter.Type.STRING:
                self.update_class = val
                mode, cls = ('remove', val[1:]) if val.startswith('-') else ('add', val)
                if cls not in self.supported_class_names:
                    result.successful = False
                    result.reason = f"'{cls}' is not a supported class name."
                    self.get_logger().warn(f"'{cls}' is not a supported class name.")
                else:
                    cls_key = self.class_names_inv.get(cls.strip())
                    if mode == 'add' and cls_key not in self.classes:
                        self.classes.append(cls_key)
                        self.get_logger().info(f"Added '{cls}' to detection classes.")
                    elif mode == 'remove' and cls_key in self.classes:
                        self.classes.remove(cls_key)
                        self.get_logger().info(f"Removed '{cls}' from detection classes.")
                    self.inference_dict['classes'] = self.classes
                    self.set_parameters([
                        Parameter(
                            'classes', Parameter.Type.STRING_ARRAY,
                            [self.class_names[x] for x in self.classes])
                    ])
            self.get_logger().info(f'Param {param.name} → {param.value}: success={result.successful}')
        return result


def main(args=None):
    rclpy.init(args=args)
    node = SingleStreamDetector()
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