import cv2
import numpy as np
from joblib import load

classifier = load("../Classifier/gesture_svm_model_4_classes.joblib")
print("Model loaded successfully")
def extract_features(binary_mask):
    features = []
    
    # Contour features
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h
        
        # Append contour-based features
        features.extend([area, perimeter, solidity, aspect_ratio])
    else:
        features.extend([0, 0, 0, 0])  # Default values for missing contours

    # Hu Moments
    moments = cv2.moments(binary_mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments)
    
    return np.array(features)


def adjust_brightness(frame):
    # Convert to YUV color space
    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv_frame)

    # Calculate the average brightness
    avg_brightness = np.mean(y_channel)

    # Normalize brightness (target brightness level)
    target_brightness = 128
    adjustment_factor = target_brightness / avg_brightness

    # Adjust the Y channel (brightness)
    y_channel = np.clip(y_channel * adjustment_factor, 0, 255).astype(np.uint8)

    # Merge channels back and convert to BGR
    adjusted_frame = cv2.merge((y_channel, u_channel, v_channel))
    return cv2.cvtColor(adjusted_frame, cv2.COLOR_YUV2BGR)

# Function to segment the hand
def segment_hand(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask for skin color
    mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)

    # Apply Gaussian Blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of the hand
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty black image
    segmented_hand = np.zeros_like(frame)

    if contours:
        # Filter contours based on area
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Adjust this threshold as needed
                cv2.drawContours(segmented_hand, [contour], -1, (255, 255, 255), thickness=cv2.FILLED)

    return cv2.cvtColor(segmented_hand, cv2.COLOR_BGR2GRAY)

# Main function to capture video and apply segmentation
def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Mirror the frame
        frame = cv2.flip(frame, 1)

        # Adjust brightness
        frame = adjust_brightness(frame)

        # Define a region of interest (ROI) on the right side of the frame
        height, width, _ = frame.shape
        roi = frame[int(height * 0.2):int(height * 0.8), int(width * 0.5):width]  # Right half

        # Segment the hand in the ROI
        hand_segmented = segment_hand(roi)
        

        # Show the original mirrored frame and the segmented hand
        features = extract_features(hand_segmented)
        prediction = classifier.predict([features])
        print(f"Predicted Gesture: {prediction[0]}")
        cv2.imshow('Original Frame', frame)
        cv2.imshow('Hand Segmentation', hand_segmented)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()