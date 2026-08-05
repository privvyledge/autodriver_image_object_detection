import numpy as np

try:
    import cv2
    CV2_INSTALLED = True
except ImportError:
    cv2 = None
    CV2_INSTALLED = False
    print("OpenCV not installed. Install by running `pip install opencv-python`")

try:
    import torch
    TORCH_INSTALLED = True
except ImportError:
    torch = None
    TORCH_INSTALLED = False

try:
    import torchvision
    TORCHVISION_INSTALLED = True
except ImportError:
    torchvision = None
    TORCHVISION_INSTALLED = False

try:
    import kornia
    KORNIA_INSTALLED = True
except ImportError:
    kornia = None
    KORNIA_INSTALLED = False
    print("OpenCV not installed. Install by running `pip install kornia kornia-rs kornia_moons`")

try:
    import skimage
    SKIMAGE_INSTALLED = True
except ImportError:
    skimage = None
    SKIMAGE_INSTALLED = False
    print("scikit-image not installed. Install by running `pip install scikit-image[optional]`")