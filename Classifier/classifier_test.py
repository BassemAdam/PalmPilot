from joblib import load
import cv2
import numpy as np

# Load the saved classifier
classifier = load("gesture_svm_model_4_classes.joblib")
print("Model loaded successfully")

# Function to extract features from a binary mask
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

# Example: Predict a gesture
binary_mask = cv2.imread("./932.jpg", cv2.IMREAD_GRAYSCALE)
if binary_mask is not None:
    # Resize image if necessary (ensure it matches training image size)
    resized_mask = cv2.resize(binary_mask, (128, 128))  # Resize to the expected size if needed
    
    # Ensure binary (0 or 255)
    _, binary_resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Extract features from the binary mask
    features = extract_features(binary_resized_mask)
    
    # Predict the gesture
    prediction = classifier.predict([features])
    print(f"Predicted Gesture: {prediction[0]}")
else:
    print("Error: Test binary mask could not be loaded.")
