import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from skimage import exposure

class GestureClassifierWithHOG:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.class_names = []
        self.model = None
        self.scaler = StandardScaler()

    def load_class_names(self, dataset_path):
        if dataset_path:
            self.dataset_path = dataset_path
            for folder_name in os.listdir(self.dataset_path):
                if os.path.isdir(os.path.join(self.dataset_path, folder_name)):
                    self.class_names.append(folder_name)

    def extract_hog_features(self, image):
        # Resize image to a fixed size (e.g., 64x64)
        image_resized = cv2.resize(image, (64, 64))

        # Compute HOG features
        fd, hog_image = hog(image_resized, pixels_per_cell=(32, 32), cells_per_block=(2, 2), visualize=True)

        # Enhance the image visualization (optional)
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return fd  # Return the HOG feature descriptor

    def load_dataset(self):
        data = []
        labels = []

        # Iterate through gesture folders
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue

            self.class_names.append(folder_name)
            class_label = len(self.class_names) - 1

            # Iterate through images in folder
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if image is None:
                        continue

                    # Extract HOG features
                    features = self.extract_hog_features(image)
                    data.append(features)
                    labels.append(class_label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        # Fit the scaler on the training data
        data = self.scaler.fit_transform(data)

        # Ensure data is 2D (n_samples, n_features)
        if data.ndim == 1:  # If there is a single feature, reshape to 2D
            data = data.reshape(-1, 1)

        return data, labels

    def train(self, data, labels):
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Train the Random Forest model
        self.model = RandomForestClassifier(n_estimators=1000, random_state=42, n_jobs=5)

        # Ensure that both training and testing data are 2D
        if X_train.ndim == 1:  # If training data is 1D, reshape to 2D
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:  # If testing data is 1D, reshape to 2D
            X_test = X_test.reshape(-1, 1)

        self.model.fit(X_train, y_train)

        # Save the trained model
        joblib.dump(self.model, "gesture_classifier_model.pkl")
        joblib.dump(self.scaler, "gesture_classifier_scaler.pkl")  # Save the scaler as well
        print("Model and scaler saved.")

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        print("Accuracy:", accuracy_score(y_test, y_pred))

    def predict(self, image):
        # Extract features from the image
        features = self.extract_hog_features(image).reshape(1, -1)  # Ensure the input is 2D

        # Normalize the features using the fitted scaler
        features = self.scaler.transform(features)  # Apply the same transformation as during training

        # Predict the gesture
        prediction = self.model.predict(features)
        probability = self.model.predict_proba(features)

        return self.class_names[prediction[0]], probability[0]

# Test function
def test_single_image(image_path):
    # Read the test image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return

    classifier = GestureClassifierWithHOG()

    # Predict the gesture and its probability
    gesture, probability = classifier.predict(image)

    # Print the result
    print(f"Predicted Gesture: {gesture}")
    print(f"Class Probabilities: {probability}")

# Main function for dataset loading and training
def main():
    dataset_path = "./test"  # Adjust the path to your dataset
    classifier = GestureClassifierWithHOG(dataset_path)

    print("Loading dataset...")
    data, labels = classifier.load_dataset()

    print("Training classifier...")
    classifier.train(data, labels)

if __name__ == "__main__":
    main()
