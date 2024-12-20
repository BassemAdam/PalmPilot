import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib
from sklearn.preprocessing import StandardScaler  # Import StandardScaler

class GestureClassifierWithHull:
    def __init__(self, dataset_path=None):
        self.dataset_path = dataset_path
        self.class_names = []
        self.model = None
        self.scaler = StandardScaler()  # Initialize the scaler

    def load_class_names(self, dataset_path):
        if dataset_path:
            self.dataset_path = dataset_path
            for folder_name in os.listdir(self.dataset_path):
                if os.path.isdir(os.path.join(self.dataset_path, folder_name)):
                    self.class_names.append(folder_name)

    def extract_hull_features(self, image):
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return np.zeros(8)  # Return empty features if no contours

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

                    # Extract convex hull features
                    features = self.extract_hull_features(image)
                    data.append(features)
                    labels.append(class_label)
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")

        # Convert to numpy arrays
        data = np.array(data)
        labels = np.array(labels)

        # Fit the scaler on the training data
        data = self.scaler.fit_transform(data)  # Fit and transform the training data

        return data, labels

    def train(self, data, labels):
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

        # Train the SVM model
        self.model = SVC(kernel='rbf', probability=True, C=1.0, gamma='scale')

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
        features = self.extract_hull_features(image).reshape(1, -1)

        # Normalize the features using the fitted scaler
        features = self.scaler.transform(features)

        # Predict the gesture
        prediction = self.model.predict(features)
        probability = self.model.predict_proba(features)

        return self.class_names[prediction[0]], probability[0]

# Main function for dataset loading and training
def main():
    dataset_path = "../Classifier/Gesture Image Pre-Processed Data"  # Use actual dataset path
    classifier = GestureClassifierWithHull(dataset_path)

    print("Loading dataset...")
    data, labels = classifier.load_dataset()

    print("Training classifier...")
    classifier.train(data, labels)

if __name__ == "__main__":
    main()
