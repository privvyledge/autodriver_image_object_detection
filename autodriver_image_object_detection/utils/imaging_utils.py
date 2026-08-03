import cv2
import numpy as np
from cv_bridge import CvBridge


def parse_image_message(
    msg,
    bridge: CvBridge,
    image_message_format: str = 'raw',
    depth_scale: float = None,
    logger=None,
):
    """Convert a ROS sensor_msgs/Image or CompressedImage to an OpenCV BGR frame.

    Args:
        msg: ROS Image or CompressedImage message.
        bridge: CvBridge instance.
        image_message_format: 'raw' | 'compressed' | 'packet'.
        depth_scale: Required only for depth images. 1000.0 for 16UC1 (mm→m),
            1.0 for 32FC1 (already metres). Pass None to skip the assertion.
        logger: Optional ROS2 logger (node.get_logger()) for error messages.

    Returns:
        10-tuple:
            (cv_image, msg_encoding, frame_id, timestamp,
             msg_fmt, conversion, inverse_conversion,
             is_color, is_depth, compressed_msg_codec)

        cv_image is always BGR (or the raw depth array for depth encodings).
        conversion / inverse_conversion are cv2 colour-code ints or None.
    """
    frame_id = msg.header.frame_id
    timestamp = msg.header.stamp
    msg_fmt = "bgr8"
    compressed_msg_codec = None
    conversion = None
    inverse_conversion = None
    is_color = True
    is_depth = False

    if image_message_format == 'compressed':
        # format string: "rgb8; jpeg compressed bgr8"
        parts = msg.format.split(';')
        codec_parts = parts[1].split()
        compressed_msg_codec = codec_parts[0]
        msg_encoding = codec_parts[-1]
    else:
        msg_encoding = msg.encoding

    if (msg_encoding.find("mono8") != -1) or (msg_encoding.find("8UC1") != -1):
        # cv_bridge refuses an 8UC1 -> mono8 conversion ("not a color format")
        # even though the buffers are byte-identical, so ask for passthrough
        # there and let the COLOR_GRAY2BGR conversion below do the expansion.
        msg_fmt = "mono8" if msg_encoding.find("mono8") != -1 else "passthrough"
        is_color = False
        conversion = cv2.COLOR_GRAY2BGR
        inverse_conversion = cv2.COLOR_BGR2GRAY
    elif msg_encoding.find("bgra") != -1:
        msg_fmt = "bgra8"
        conversion = cv2.COLOR_BGRA2BGR
        inverse_conversion = cv2.COLOR_BGR2BGRA
    elif msg_encoding.find("rgba") != -1:
        msg_fmt = "rgba8"
        conversion = cv2.COLOR_RGBA2BGR
        inverse_conversion = cv2.COLOR_BGR2RGBA
    elif msg_encoding.find("bgr8") != -1:
        msg_fmt = "bgr8"
    elif msg_encoding.find("rgb8") != -1:
        msg_fmt = "rgb8"
        conversion = cv2.COLOR_RGB2BGR
        inverse_conversion = cv2.COLOR_BGR2RGB
    elif msg_encoding.find("16UC1") != -1:
        msg_fmt = "16UC1"
        is_color = False
        is_depth = True
        if depth_scale is not None:
            assert depth_scale == 1000.0, (
                f"16UC1 depth is in mm — depth_scale must be 1000.0, got {depth_scale}"
            )
    elif msg_encoding.find("32FC1") != -1:
        msg_fmt = "32FC1"
        is_color = False
        is_depth = True
        if depth_scale is not None:
            assert depth_scale == 1.0, (
                f"32FC1 depth is in metres — depth_scale must be 1.0, got {depth_scale}"
            )
    else:
        if logger is not None:
            logger.error(f"parse_image_message: unsupported encoding '{msg_encoding}'")
        else:
            print(f"[imaging_utils] unsupported encoding: {msg_encoding}")
        msg_fmt = 'passthrough'

    if image_message_format in ("compressed", "packet"):
        cv_image = bridge.compressed_imgmsg_to_cv2(msg, msg_fmt)
    else:
        cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding=msg_fmt)

    if conversion is not None:
        cv_image = cv2.cvtColor(cv_image, conversion)

    return (
        cv_image, msg_encoding, frame_id, timestamp,
        msg_fmt, conversion, inverse_conversion,
        is_color, is_depth, compressed_msg_codec,
    )