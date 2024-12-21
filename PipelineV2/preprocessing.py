# preprocessing.py
import cv2
import numpy as np

class ImagePreprocessor:
    def __init__(self):
        self.kernel = np.ones((3,3), np.uint8)
        
    def process(self, frame):
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Add noise reduction if needed
        # denoised = cv2.GaussianBlur(rgb_frame, (5,5), 0)
        return rgb_frame