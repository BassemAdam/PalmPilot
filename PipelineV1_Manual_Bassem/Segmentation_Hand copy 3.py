import cv2
import numpy as np
import time
from collections import deque

class AdvancedHandSegmenter:
    def __init__(self, adaptive_thresholds=True, history_size=5009, bg_history=3000, var_threshold=5, detect_shadows=False):
        self.adaptive_thresholds = adaptive_thresholds
        self.prev_frame = None
        self.contour_history = deque(maxlen=history_size)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.prev_gray = None  # For optical flow
        self.n_colors = 8  # Number of colors for quantization
        self.hand_center = None
        self.protection_radius = 100  # Radius of protected region
        self.mask_history = deque(maxlen=10)
    
    def cartoonize_frame(self, frame, mask):
        # K-means clustering with single skin color
        data = frame[mask > 0].reshape((-1,3))
        if len(data) == 0:
            return np.zeros_like(frame)
        
        data = np.float32(data)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(data, 1, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Get dominant skin color
        skin_color = np.uint8(centers[0])
        
        # Create result frame
        result = np.zeros_like(frame)
        result[mask > 0] = skin_color
        
        # Verify skin color in HSV space
        hsv_result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv_result, lower_skin, upper_skin)
        
        # Apply final mask
        result[skin_mask == 0] = 0
        
        return result
    
    def get_hand_center(self, contour):
        if contour is None:
            return None
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)
        return None

    def create_circle_mask(self, frame_shape, center, radius):
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        if center is not None:
            cv2.circle(mask, center, radius, 255, -1)
        return mask

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
        
        combined_mask = cv2.addWeighted(mask_hsv, 0.8, mask_ycrcb, 0.5, 0)
        # fg = self.bg_subtractor.apply(frame)
        # combined_mask = cv2.bitwise_and(combined_mask, fg)
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
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            return np.zeros_like(gray)
        
        flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        self.prev_gray = gray
        return mag

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
            
            fg_mask = self.bg_subtractor.apply(frame)
            combined_mask_fg_mask = cv2.bitwise_and(skin_mask, fg_mask)
        
            _, motion_mask = cv2.threshold(optical_flow_mask, 1.0, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.uint8)
            
            combined_mask = cv2.bitwise_and(skin_mask, motion_mask)

          

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
                    
                    # Get hand center
                    self.hand_center = self.get_hand_center(smoothed_contour)
                    
                    # Create protected region mask
                    protection_mask = self.create_circle_mask(frame.shape, self.hand_center, self.protection_radius)
                    
                    
                    # Create result frame with protected region
                    result = frame.copy()
                    protection_mask = self.create_circle_mask(frame.shape, self.hand_center, self.protection_radius)
                    result[protection_mask == 0] = [0, 0, 0]  # Mask everything outside protected region
                    
                    # Draw protected region circle
                    if self.hand_center:
                        cv2.circle(result, self.hand_center, self.protection_radius, (0,255,0), 2)
                    
                    self.mask_history.append(protection_mask)
                    
                    # Add hand features visualization only within protected region
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
                                # Only draw features within protected region
                                if protection_mask[start[1], start[0]] > 0:
                                    cv2.circle(result, start, 5, (255,0,0), -1)  # Blue fingertips
                                if protection_mask[end[1], end[0]] > 0:
                                    cv2.circle(result, end, 5, (255,0,0), -1)
                                if protection_mask[far[1], far[0]] > 0:
                                    cv2.circle(result, far, 5, (255,255,0), -1)  # Cyan valleys
                    
                    combined_mask = cv2.bitwise_and(combined_mask, protection_mask)
                    return combined_mask, combined_mask
                    
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