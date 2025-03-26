"""
Subscribes to the image and detection result topics using a MessageFilter, then plots the detections.
Uses a FIFO queue to store the last n detections and plots them on the image.
The image is then published to a new topic.

Todo: See my LUSCIO_ROS repo for a sample ROS1 implementation.
"""