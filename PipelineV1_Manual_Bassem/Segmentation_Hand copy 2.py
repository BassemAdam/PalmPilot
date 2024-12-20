import cv2
import numpy as np
import time
from collections import deque
import collections
class AdvancedHandSegmenter:
    def __init__(self, adaptive_thresholds=True, history_size=300, bg_history=300, var_threshold=5, detect_shadows=False):
        self.adaptive_thresholds = adaptive_thresholds
        self.prev_frame = None
        self.contour_history = deque(maxlen=history_size)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.prev_gray = None  # For optical flow
        self.n_colors = 8  # Number of colors for quantization
        self.hand_center = None
        self.protection_radius = 100  # Radius of protected region
        self.mask_history = deque(maxlen=100)
        self.prev_mag = None
        self.motion_history = None
        self.MHI_DURATION = 0.5
        self.MAX_TIME_DELTA = 0.25
        self.MIN_TIME_DELTA = 0.05
    def cartoonize_frame(self, frame, mask):
        # Edge-aware smoothing
        smooth = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 9, 2)
        
        # Color quantization
        data = smooth[mask > 0].reshape((-1,3))
        if len(data) == 0:
            return np.zeros_like(frame)
        
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
        K = 8  # Number of colors
        _, labels, centers = cv2.kmeans(data, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Create cartoon effect
        result = np.zeros_like(frame)
        labels = labels.flatten()
        for i, color in enumerate(centers):
            result[mask > 0][labels == i] = color
        
        # Combine with edges
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.bitwise_and(result, 255 - edges)
        
        # Apply mask
        result[mask == 0] = 0
        
        cv2.imshow('Cartoon Effect', result)
        return result
    
    def create_enhanced_skin_mask(self, frame):
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        
        # Adjust thresholds based on brightness
        if avg_brightness < 85:  # Dark
            lower_hsv = np.array([0, 25, 50], dtype=np.uint8)
            upper_hsv = np.array([25, 160, 255], dtype=np.uint8)
            lower_ycrcb = np.array([0, 130, 80], dtype=np.uint8)
            upper_ycrcb = np.array([255, 175, 130], dtype=np.uint8)
        elif avg_brightness > 170:  # Bright
            lower_hsv = np.array([0, 35, 70], dtype=np.uint8)
            upper_hsv = np.array([15, 140, 255], dtype=np.uint8)
            lower_ycrcb = np.array([0, 140, 90], dtype=np.uint8)
            upper_ycrcb = np.array([255, 185, 140], dtype=np.uint8)
        else:  # Normal lighting
            lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
            upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
            lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
            upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        combined_mask = cv2.addWeighted(mask_hsv, 0.5, mask_ycrcb, 0.5, 0)
     
        return combined_mask

    def is_hand_contour(self, contour):
        # Get basic contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        
        # Calculate shape features
        solidity = float(area)/hull_area if hull_area > 0 else 0
        x,y,w,h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h if h > 0 else 0
        
        # Hand typically has:
        # - Lower solidity (0.6-0.9) due to fingers
        # - Aspect ratio between 0.5-1.5
        # - Significant convexity defects
        is_hand = (
            solidity > 0.6 and solidity < 0.9 and
            aspect_ratio > 0.5 and aspect_ratio < 1.5 and
            area > 3000
        )
        
        return is_hand

    
    def calculate_optical_flow(self, frame):
        # Face detection to exclude face region
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
        
        # Create face exclusion mask
        face_mask = np.ones_like(gray_frame)
        for (x, y, w, h) in faces:
            face_mask[y:y+h, x:x+w] = 0
        
        # Create skin mask with face excluded
        skin_mask = self.create_enhanced_skin_mask(frame)
        skin_mask = cv2.bitwise_and(skin_mask, skin_mask, mask=face_mask)
        
        # Preprocessing with skin mask
        blurred = cv2.GaussianBlur(frame, (7, 7), 1.5)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        masked_gray = cv2.bitwise_and(gray, gray, mask=skin_mask)
        
        # Initialize tracking
        if self.prev_gray is None:
            self.prev_gray = masked_gray.copy()
            self.prev_mag = np.zeros_like(masked_gray, dtype=np.float32)
            self.motion_history = np.zeros_like(masked_gray, dtype=np.float32)
            self.hand_roi = None
            return np.zeros_like(masked_gray)
    
        # Calculate optical flow only in ROI if available
        if self.hand_roi is not None:
            x, y, w, h = self.hand_roi
            roi_mask = np.zeros_like(masked_gray)
            roi_mask[y:y+h, x:x+w] = 1
            masked_gray = cv2.bitwise_and(masked_gray, masked_gray, mask=roi_mask)
    
        # Enhanced optical flow
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            masked_gray,
            None,
            pyr_scale=0.5,
            levels=6,
            winsize=21,
            iterations=5,
            poly_n=7,
            poly_sigma=1.5,
            flags=0
        )
    
        # Motion analysis
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Adaptive thresholding for hand movement
        mag_threshold = np.percentile(mag, 85)  # More strict threshold
        hand_motion_mask = (mag > mag_threshold).astype(np.uint8) * 255
        
        # Find the largest connected component (assumed to be the hand)
        contours, _ = cv2.findContours(hand_motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            # Filter by minimum area to avoid small movements
            if cv2.contourArea(largest_contour) > 1000:
                hand_motion_mask = np.zeros_like(hand_motion_mask)
                cv2.drawContours(hand_motion_mask, [largest_contour], -1, 255, -1)
                # Update ROI
                self.hand_roi = cv2.boundingRect(largest_contour)
        
        # Update states
        self.prev_gray = masked_gray.copy()
        self.prev_mag = mag.astype(np.float32)
    
        return cv2.bitwise_and(hand_motion_mask, hand_motion_mask, mask=skin_mask)
    
    
    def smooth_contours(self, contour):
        try:
            perimeter = cv2.arcLength(contour, True)
            complexity_of_contour = 0.1
            epsilon = complexity_of_contour * perimeter
            approx_contour = cv2.approxPolyDP(contour, epsilon, True)
            
            if len(approx_contour) < 20:
                return contour
                
            return approx_contour
        except:
            return contour
    
    def analyze_hand_features(self, contour, frame):
        # Get convex hull
        hull = cv2.convexHull(contour)
        hull_indices = cv2.convexHull(contour, returnPoints=False)
        
        # Get defects
        defects = cv2.convexityDefects(contour, hull_indices)
        
        # Create visualization
        result = frame.copy()
        
        # Draw contour (red)
        cv2.drawContours(result, [contour], -1, (0,0,255), 2)
        
        # Draw hull (green)
        cv2.drawContours(result, [hull], -1, (0,255,0), 2)
        
        if defects is not None:
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                
                # Filter significant defects
                if d > 10000:
                    # Draw fingertips (blue)
                    cv2.circle(result, start, 5, (255,0,0), -1)
                    cv2.circle(result, end, 5, (255,0,0), -1)
                    # Draw valleys (cyan)
                    cv2.circle(result, far, 5, (255,255,0), -1)
        
        return result


    def segment_hand(self, frame):
        try:
            enhanced_frame = frame.copy()
            skin_mask = self.create_enhanced_skin_mask(enhanced_frame)
            optical_flow_mask = self.calculate_optical_flow(enhanced_frame)
            
           
            _, motion_mask = cv2.threshold(optical_flow_mask, 1.0, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.uint8)
            
            combined_mask = cv2.bitwise_and(skin_mask, motion_mask)

            # fg_mask = self.bg_subtractor.apply(frame)
            # combined_mask_fg_mask = cv2.bitwise_and(skin_mask, fg_mask)
        

            # Add edge detection
            edges = cv2.Canny(frame, 100, 200)
            combined_mask = cv2.bitwise_or(combined_mask, edges)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                hand_contours = [cnt for cnt in contours if self.is_hand_contour(cnt)]
                
                if hand_contours:
                    max_contour = max(hand_contours, key=cv2.contourArea)
                    smoothed_contour = self.smooth_contours(max_contour)
                
                    self.contour_history.append(smoothed_contour)
                    
                    # Create white hand mask
                    hand_mask = np.zeros_like(skin_mask)
                    cv2.drawContours(hand_mask, [smoothed_contour], -1, 255, -1)
                    
                    # Create white hand region with feature visualization
                    result = np.zeros_like(frame)
                    result[hand_mask > 0] = [255, 255, 255]  # Make hand region white
                    
                    # Add feature visualization on white background
                    cv2.drawContours(result, [smoothed_contour], -1, (0,0,255), 2)  # Red contour
                    
                    # Get hull and defects for visualization
                    hull = cv2.convexHull(smoothed_contour)
                    hull_indices = cv2.convexHull(smoothed_contour, returnPoints=False)
                    cv2.drawContours(result, [hull], -1, (0,255,0), 2)  # Green hull
                    
                    defects = cv2.convexityDefects(smoothed_contour, hull_indices)
                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s,e,f,d = defects[i,0]
                            if d > 10000:
                                start = tuple(smoothed_contour[s][0])
                                end = tuple(smoothed_contour[e][0])
                                far = tuple(smoothed_contour[f][0])
                                cv2.circle(result, start, 5, (255,0,0), -1)  # Blue fingertips
                                cv2.circle(result, end, 5, (255,0,0), -1)
                                cv2.circle(result, far, 5, (255,255,0), -1)  # Cyan valleys
                    
                    cv2.imshow("Hand Features", result)
                    return result, hand_mask
                    
            return frame, combined_mask
        
        except Exception as e:
            print(f"Error in hand segmentation: {str(e)}")
            return frame, np.zeros_like(frame[:,:,0])



def main():
    # Video file path
    video_path = r"C:\Users\basim\Desktop\Test2.mp4"
    
    cap = cv2.VideoCapture(0)
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