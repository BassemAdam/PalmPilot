import cv2
import numpy as np
import time
from collections import deque
from classifier_testV2 import test_single_image

class AdvancedHandSegmenter:
    
    def __init__(self, adaptive_thresholds=True, history_size=3000, bg_history=3000, var_threshold=10, detect_shadows=False):
        self.adaptive_thresholds = adaptive_thresholds
        self.prev_frame = None
        self.contour_history = deque(maxlen=history_size)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=var_threshold, detectShadows=detect_shadows)
        self.prev_gray = None  # For optical flow
        self.n_colors = 8  # Number of colors for quantization
        self.hand_center = None
        self.protection_radius = 100  # Radius of protected region
        self.mask_history = deque(maxlen=10)
    
    def exclude_face_region(self, frame, mask):
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Create a mask to exclude the face
        for (x, y, w, h) in faces:
            # Calculate center of face
            center_x = x + w//2
            center_y = y + h//2
            
            # Increase dimensions by 50% for padding
            radius = int(max(w, h) * 0.9)
            
            # Draw filled circle to mask out expanded face region
            cv2.circle(mask, (center_x, center_y), radius, 0, -1)
            
            # Optional: Draw rectangle for visualization
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
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
         # Check number of channels
        if len(frame.shape) == 3:  # Color image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:  # Already grayscale
            gray = frame
        
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

    
    def process_segments(self,image):
        # Convert the image to HSV and YCrCb color spaces
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        
        # Optimized thresholds for skin detection
        lower_hsv = np.array([0, 30, 60], dtype=np.uint8)
        upper_hsv = np.array([20, 150, 255], dtype=np.uint8)
        lower_ycrcb = np.array([0, 135, 85], dtype=np.uint8)
        upper_ycrcb = np.array([255, 180, 135], dtype=np.uint8)
        
        mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
        mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
        
        # Combine masks with weights
        combined_mask = cv2.addWeighted(mask_hsv, 0.5, mask_ycrcb, 0.5, 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        segmented_image = image.copy()

        if len(contours) >= 2:
            largest = contours[0]
            second_largest = contours[1]
        elif len(contours) == 1:
            x, y, w, h = cv2.boundingRect(contours[0])
            top_half = contours[0][contours[0][:, 0, 1] < y + h // 2]
            bottom_half = contours[0][contours[0][:, 0, 1] >= y + h // 2]
            if top_half.shape[0] > 0 and bottom_half.shape[0] > 0:
                largest = top_half
                second_largest = bottom_half
            else:
                return image, None, None, None
        else:
            return image, None, None, None

        # Debugging: Print contour shapes
        #print(f"Largest Contour: {largest.shape}")
        #print(f"Second Largest Contour: {second_largest.shape}")

        # Draw red and blue regions on the segmented image
        if len(largest) > 0:
            cv2.drawContours(segmented_image, [largest], -1, (255, 0, 0), -1)  # Blue for the largest
        if len(second_largest) > 0:
            cv2.drawContours(segmented_image, [second_largest], -1, (0, 0, 255), -1)  # Red for the second largest

        def resize_with_aspect_ratio(image, target_size=(300, 300), pad_color=0):
            h, w = image.shape[:2]
            target_w, target_h = target_size
            scale = min(target_w / w, target_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            canvas = np.full((target_h, target_w), pad_color, dtype=np.uint8)
            x_offset = (target_w - new_w) // 2
            y_offset = (target_h - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
            return canvas

        def create_segment(contour, target_size=(300, 300)):
            if contour is None or len(contour) == 0:
                return None
            mask = np.zeros_like(image[:, :, 0])
            cv2.drawContours(mask, [contour], -1, 255, -1)
            x, y, w, h = cv2.boundingRect(contour)
            cropped = mask[y:y + h, x:x + w]
            return resize_with_aspect_ratio(cropped, target_size)

        blue_segment = create_segment(largest)
        red_segment = create_segment(second_largest)

        return image, segmented_image, blue_segment, red_segment


    def segment_hand(self, frame):
        try:
            enhanced_frame = frame.copy()
          
            skin_mask = self.create_enhanced_skin_mask(enhanced_frame)
      
            skin_mask = self.exclude_face_region(enhanced_frame, skin_mask)
            #cv2.imshow("skin_mask skin_mask", skin_mask)
            
            optical_flow_mask = self.calculate_optical_flow(skin_mask)
            #cv2.imshow("optical_flow_mask", optical_flow_mask)
            
            fg_mask = self.bg_subtractor.apply(frame)
            #cv2.imshow("fg_mask", fg_mask)
            
            combined_mask_fg_mask = cv2.bitwise_and(skin_mask, fg_mask)
            cv2.imshow("combined_mask_fg_mask", combined_mask_fg_mask)
            
            _, motion_mask = cv2.threshold(optical_flow_mask, 1.0, 255, cv2.THRESH_BINARY)
            motion_mask = motion_mask.astype(np.uint8)
                      # Replace line with noise removal pipeline
            kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            combined_mask = cv2.bitwise_and(skin_mask, combined_mask_fg_mask)
            
         
            
            # Add edge detection
            edges = cv2.Canny(frame, 200, 250)
            combined_mask = cv2.bitwise_or(combined_mask, edges)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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
    video_path = r"C:\Users\basim\Desktop\Test1.mp4"
    
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
        
   
    
    
        test_single_image(mask)
        cv2.imshow("Original", frame)
        cv2.imshow("Segmented Hand", result)
        cv2.imshow("Mask", mask)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()