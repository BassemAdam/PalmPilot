import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage import exposure

# Load the saved model and scaler
model = joblib.load("gesture_classifier_model.pkl")
scaler = joblib.load("gesture_classifier_scaler.pkl")

class GestureClassifierWithHOG:
    def __init__(self):
        self.class_names = ['A', 'B', 'C', 'D']  # Replace with actual class names

    def extract_hog_features(self, image):
        # Resize image to a fixed size (e.g., 64x64)
        image_resized = cv2.resize(image, (64, 64))

        # Compute HOG features
        fd, hog_image = hog(image_resized, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=True)

        # Enhance the image visualization (optional)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return fd  # Return the HOG feature descriptor

    def predict(self, image):
        # Extract features from the image
        features = self.extract_hog_features(image).reshape(1, -1)  # Ensure the input is 2D

        # Normalize the features using the fitted scaler
        features = scaler.transform(features)  # Apply the same transformation as during training

        # Predict the gesture
        prediction = model.predict(features)
        probability = model.predict_proba(features)

        return self.class_names[prediction[0]], probability[0]

# Test function
def test_single_image(image):
    # Read the test image
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded successfully
    classifier = GestureClassifierWithHOG()

    # Predict the gesture and its probability
    gesture, probability = classifier.predict(image)

    # Print the result
    print(f"Predicted Gesture: {gesture}")
    print(f"Class Probabilities: {probability}")

# Run the test after the model is trained
if __name__ == "__main__":
    # Ensure that the model and scaler are trained first
    # You can call main() function here if not already done
    # main()  # Uncomment if model needs to be trained
    a = cv2.imread("hand_binary_mask.jpg", cv2.IMREAD_GRAYSCALE)
    test_single_image(a)  # Provide a test image path
