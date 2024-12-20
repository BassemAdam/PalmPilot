import cv2
import numpy as np
import time
from collections import deque
import logging
from pathlib import Path
from tqdm import tqdm

class EnhancedHandSegmenter:
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'history_size': 10,
            'bg_history': 500,
            'min_area': 3000,
            'blur_kernel': (7, 7),
            'skin_thresh': {
                'hsv': {'lower': [0, 20, 70], 'upper': [20, 170, 255]},
                'ycrcb': {'lower': [0, 135, 85], 'upper': [255, 180, 135]}
            }
        }
        if config:
            self.config.update(config)

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config['bg_history'],
            varThreshold=25,
            detectShadows=True
        )
        self.contour_history = deque(maxlen=self.config['history_size'])
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def detect_fingers(self, contour, hull):
        """Improved finger detection using angle-based approach"""
        if len(contour) < 5:
            return 0

        hull_points = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull_points)
        
        if defects is None:
            return 0
            
        finger_count = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            
            # Enhanced angle calculation
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            
            # Avoid division by zero
            if b * c == 0:
                continue
                
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c))
            
            # Refined angle threshold
            if angle <= np.pi / 2.5 and d > 30000:  # Added depth threshold
                finger_count += 1
                
        return min(finger_count + 1, 5)  # Cap at 5 fingers

def main():
    # Use pathlib for cross-platform path handling
    # video_path = Path(input("Enter video path: ").strip('"'))
    # if not video_path.exists():
    #     logging.error(f"Video file not found: {video_path}")
    #     return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # logging.error(f"Failed to open video: {video_path}")
        return

    # Setup video writer
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # output_path = video_path.with_name(f"{video_path.stem}_processed.mp4")
    # out = cv2.VideoWriter(
    #     str(output_path),
    #     cv2.VideoWriter_fourcc(*'mp4v'),
    #     30,
    #     (frame_width, frame_height)
    # )

    segmenter = EnhancedHandSegmenter()
    
    # Process with progress bar
    with tqdm(total=total_frames) as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result, mask = segmenter.segment_hand(frame)
            # out.write(result)
            
            # Display results
            cv2.imshow("Processed", result)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            pbar.update(1)

    cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()