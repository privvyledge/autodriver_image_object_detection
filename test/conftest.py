"""Put the repository root on sys.path so `autodriver_image_object_detection.*` imports resolve.

This package directory is the repository root, and the Python package lives one
level down in `autodriver_image_object_detection/`. Tests run without a built
workspace, so the path is inserted here rather than relying on `install/`.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
