"""
Todo:
    * Test displaying ROIs, e.g static
    * Mouse callback for ROI selection
    * Implement add/remove ROI methods and test with the mouse callback
    * Add adding/removing classes and test with the mouse callback
    * Test adjusting ROIs (moving, resizing)
    * Implement private method to record click to use for testing intersection
    * Intersection tests (point in polygon, IoU, convex hull intersection)
    * Implement detection class (e.g similar to asOneDetector for future modularity)
    * Implement detection in ROIs
    * Implement detection in ROIs using masking, i.e bitwise_and
    * Implement counting in ROIs
    * Implement tracking in ROIs
    * Implement separate tracking for objects in ROIs instead of tracking across the entire image
    * Setup Torch interface
    * Setup batch processing
    * Create a ROS2 node
    * Switch CV2 to Transparent API
"""
import numpy as np
import sklearn
import shapely
import cv2
from ultralytics import YOLO

try:
    import torch, torch.utils.dlpack, torchvision
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ROIBasedDetector:
    def __init__(self):
        self.prefer_torch = TORCH_AVAILABLE
        self.frame = None
        self.roi = None  # list of ROI coordinates
        self.model = YOLO('yolov8n.pt')


