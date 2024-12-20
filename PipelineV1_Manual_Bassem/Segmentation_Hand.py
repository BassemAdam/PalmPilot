import cv2
import numpy as np
import time
from collections import deque

class AdvancedHandSegmenter:
    def __init__(self, adaptive_thresholds=True, history_size=5):
        self.adaptive_thresholds = adaptive_thresholds
        self.prev_frame = None
        self.contour_history = deque(maxlen=history_size)
        # Background subtractor parameters
        sensitivity_motion = 3
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=sensitivity_motion, detectShadows=False)
        
    def create_enhanced_skin_mask(self, frame):
        # Improved color thresholds for better skin detection
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        
        # Optimized thresholds for skin detection
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks with improved weights
        combined_mask = cv2.addWeighted(mask_hsv, 0.5, mask_ycrcb, 0.5, 0)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        combined_mask = cv2.bitwise_and(combined_mask, fg_mask)
        
        return combined_mask

    def smooth_contours(self, contour):
        try:
            # Interpolate points to have consistent number of points
            perimeter = cv2.arcLength(contour, True)
            complexity_of_contour = 0.1  # Smaller value for smoother contours
            epsilon = complexity_of_contour * perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            # Ensure minimum number of points
            if len(approx_contour) < 20:
                return contour
                
            return approx_contour
        except:
            return contour

    
    def segment_hand(self, frame):
        try:
            # Create skin mask
            skin_mask = self.create_enhanced_skin_mask(frame)
            
            # Enhanced morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find and filter contours
            """
            its important that you understand the other parameters so i will write them here for the discussion
            we can also keep playing with them but the ones that i choose worked fine so dont change it :)
            mode (second parameter)
            cv2.RETR_EXTERNAL: Only retrieves outer contours
            Other options:
            cv2.RETR_LIST: Retrieves all contours
            cv2.RETR_TREE: Retrieves all contours in hierarchy
            cv2.RETR_CCOMP: Retrieves 2-level hierarchy
            method (third parameter)

            cv2.CHAIN_APPROX_SIMPLE: Compresses horizontal, vertical, diagonal segments
            Other options:
            cv2.CHAIN_APPROX_NONE: Stores all contour points
            cv2.CHAIN_APPROX_TC89_L1: Applies Teh-Chin chain approximation
            cv2.CHAIN_APPROX_TC89_KCOS: Another Teh-Chin approximation
            """
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get largest contour assumed its the hand contour
                max_contour = max(contours, key=cv2.contourArea)
                
                if cv2.contourArea(max_contour) > 3000:  # Minimum area threshold
                    # Smooth contour
                    smoothed_contour = self.smooth_contours(max_contour)
                    self.contour_history.append(smoothed_contour)
                    
                    # Create result mask
                    hand_mask = np.zeros_like(skin_mask)
                    cv2.drawContours(hand_mask, [smoothed_contour], -1, 255, -1)
                    
                    # Apply mask to original frame
                    result = cv2.bitwise_and(frame, frame, mask=hand_mask)
                    return result, hand_mask
            
            return frame, skin_mask
            
        except Exception as e:
            print(f"Error in hand segmentation: {str(e)}")
            return frame, np.zeros_like(frame[:,:,0])


def main():
    # Video file path
    video_path = r"C:\Users\basim\Desktop\Test1.mp4"
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
        
    segmenter = AdvancedHandSegmenter()
    
    # Get video properties and optimize
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = 2  # Process every nth frame
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
            
        # Resize frame for faster processing
        height = int(frame.shape[0] * 0.5)
        width = int(frame.shape[1] * 0.5)
        frame = cv2.resize(frame, (width, height))
        
        frame = cv2.flip(frame, 1)
        result, mask = segmenter.segment_hand(frame)
        
        cv2.imshow("Original", frame)
        cv2.imshow("Segmented Hand", result)
        cv2.imshow("Mask", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()