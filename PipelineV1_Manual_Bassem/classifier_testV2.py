import cv2
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("gesture_classifier_model.pkl")
scaler = joblib.load("gesture_classifier_scaler.pkl")

# Define the classifier with the same features extraction process
class GestureClassifierWithHull:
    def __init__(self):
        self.class_names = ['A', 'B', 'C', 'D']  # Replace with actual class names

    def extract_hull_features(self, image):
        # Preprocess the image
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(8)  # Return empty features if no contours (adjusted for new number of features)

        # Get the largest contour (assume it's the gesture)
        max_contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(max_contour)

        # Extract geometric features
        area = cv2.contourArea(max_contour)
        hull_area = cv2.contourArea(hull)
        perimeter = cv2.arcLength(max_contour, True)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        x, y, w, h = cv2.boundingRect(max_contour)
        aspect_ratio = float(w) / h if h > 0 else 0

        # Extract Hu Moments
        moments = cv2.moments(max_contour)
        hu_moments = cv2.HuMoments(moments).flatten()

        # Extract Convex Hull Defects (useful for shapes like a "peace" sign)
        hull_defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False))
        defects_count = 0
        if hull_defects is not None:
            defects_count = len(hull_defects)

        # Combine all features into one array
        features = np.hstack([area, hull_area, perimeter, solidity, aspect_ratio, hu_moments, defects_count])

        return features

    def predict(self, image):
        # Extract features from the image
        features = self.extract_hull_features(image).reshape(1, -1)

        # Normalize the features using the fitted scaler
        features = scaler.transform(features)  # Apply the same transformation as during training

        # Predict the gesture
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return self.class_names[prediction[0]], probability[0]

# Test function
def test_single_image(image_path):
    # Read the test image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    classifier = GestureClassifierWithHull()

    # Predict the gesture and its probability
    gesture, probability = classifier.predict(image)

    # Print the result
    print(f"Predicted Gesture: {gesture}")
    print(f"Class Probabilities: {probability}")

# Call the test function with a test image
image_path = "segmented_hand.png"  # Replace with the actual path to your test image
test_single_image(image_path)
