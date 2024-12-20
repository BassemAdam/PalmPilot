import cv2
import numpy as np
from collections import deque

class AdvancedHandSegmenter:
    def __init__(self, adaptive_thresholds=True, history_size=5,
                 bg_history=300, var_threshold=2, detect_shadows=False, 
                 bg_ratio=1, n_mixtures=5, shadow_value=127, shadow_threshold=0.5):
        
        self.adaptive_thresholds = adaptive_thresholds
        self.prev_frame = None
        self.contour_history = deque(maxlen=history_size)
        # Background subtractor parameters
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_history,
                                                                varThreshold=var_threshold,
                                                                detectShadows=detect_shadows)
        self.bg_subtractor.setBackgroundRatio(bg_ratio)
        self.bg_subtractor.setNMixtures(n_mixtures)
        self.bg_subtractor.setShadowValue(shadow_value)
        self.bg_subtractor.setShadowThreshold(shadow_threshold)
        
        # Add mask history for temporal smoothing
        self.mask_history = deque(maxlen=history_size)
        self.frame_count = 0
        
    def detect_motion(self, frame):
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.zeros_like(self.prev_frame)
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        self.prev_frame = gray
        return motion_mask

    def detect_edges(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        return edges

    def create_enhanced_skin_mask(self, frame):
        # Apply histogram equalization to handle different lighting
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l,a,b))
        enhanced_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # Convert to different color spaces
        blurred = cv2.GaussianBlur(enhanced_frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        
        # Multiple ranges for different skin tones
        # Light to dark skin tones in HSV
        lower_hsv1 = np.array([0, 20, 50], dtype=np.uint8)
        upper_hsv1 = np.array([25, 255, 255], dtype=np.uint8)
        lower_hsv2 = np.array([165, 20, 50], dtype=np.uint8)
        upper_hsv2 = np.array([180, 255, 255], dtype=np.uint8)
        
        # YCrCb ranges for multiple skin tones
        lower_ycrcb = np.array([0, 130, 75], dtype=np.uint8)
        upper_ycrcb = np.array([255, 185, 145], dtype=np.uint8)
        
        # Create masks
        mask_hsv1 = cv2.inRange(hsv, lower_hsv1, upper_hsv1)
        mask_hsv2 = cv2.inRange(hsv, lower_hsv2, upper_hsv2)
        mask_hsv = cv2.bitwise_or(mask_hsv1, mask_hsv2)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks with adaptive weights
        mean_brightness = np.mean(l)
        hsv_weight = min(1.0, max(0.3, mean_brightness / 255))
        ycrcb_weight = 1 - hsv_weight
        
        combined_mask = cv2.addWeighted(mask_hsv, hsv_weight, 
                                      mask_ycrcb, ycrcb_weight, 0)
        
        # Background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        combined_mask = cv2.bitwise_and(combined_mask, fg_mask)
        
        # Adaptive thresholding based on lighting
        if self.adaptive_thresholds:
            mean_val = np.mean(combined_mask)
            threshold = max(0, min(255, mean_val + 30))
            _, combined_mask = cv2.threshold(combined_mask, threshold, 
                                          255, cv2.THRESH_BINARY)
        
        return combined_mask

    def analyze_shape(self, contour):
        if len(contour) < 5:
            return False
            
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(contour)
        
        if hull_area == 0:
            return False
            
        solidity = float(contour_area) / hull_area
        
        # Aspect ratio
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        # Check if shape matches typical hand characteristics
        return 0.5 < solidity < 0.95 and 0.5 < aspect_ratio < 2.0

    
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
            # 1. Motion Detection
            motion_mask = self.detect_motion(frame)
            
            # 2. Edge Detection
            edges = self.detect_edges(frame)
            
            # 3. Skin Detection
            skin_mask = self.create_enhanced_skin_mask(frame)
            
            # 4. Combine all masks
            combined_mask = cv2.bitwise_and(skin_mask, motion_mask)
            combined_mask = cv2.bitwise_or(combined_mask, edges)
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, 
                                         cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for contour in contours:
                if cv2.contourArea(contour) > 3000 and self.analyze_shape(contour):
                    valid_contours.append(contour)
            
            if valid_contours:
                max_contour = max(valid_contours, key=cv2.contourArea)
                smoothed_contour = self.smooth_contours(max_contour)
                self.contour_history.append(smoothed_contour)
                
                hand_mask = np.zeros_like(combined_mask)
                cv2.drawContours(hand_mask, [smoothed_contour], -1, 255, -1)
                
                result = cv2.bitwise_and(frame, frame, mask=hand_mask)
                return result, hand_mask
            
            return frame, combined_mask
            
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