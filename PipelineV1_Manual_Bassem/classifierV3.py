import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

class GestureClassifierWithBoVW:
    def __init__(self, dataset_path=None, num_clusters=50):
        self.dataset_path = dataset_path
        self.class_names = []
        self.num_clusters = num_clusters
        self.model = None
        self.scaler = StandardScaler()
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        self.visual_words = None

    def load_class_names(self, dataset_path):
        """Load class names from dataset directory."""
        if dataset_path:
            self.dataset_path = dataset_path
            for folder_name in os.listdir(self.dataset_path):
                if os.path.isdir(os.path.join(self.dataset_path, folder_name)):
                    self.class_names.append(folder_name)

    def extract_sift_features(self, image):
        """Extract SIFT descriptors from an image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.sift.detectAndCompute(gray, None)
        if descriptors is None:
            return np.array([])
        return descriptors

    def create_bovw_histogram(self, descriptors):
        """Create a histogram of visual words from SIFT descriptors."""
        if descriptors.size == 0:
            return np.zeros(self.num_clusters)

        labels = self.kmeans.predict(descriptors)
        histogram, _ = np.histogram(labels, bins=np.arange(self.num_clusters + 1))
        return histogram

    def load_dataset(self):
        """Load dataset, extract features, and prepare visual words."""
        sift_descriptors = []
        image_histograms = []
        labels = []

        for class_name in self.class_names:
            class_folder = os.path.join(self.dataset_path, class_name)
            for image_name in os.listdir(class_folder):
                image_path = os.path.join(class_folder, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    descriptors = self.extract_sift_features(image)
                    sift_descriptors.append(descriptors)
                    labels.append(class_name)

        # Combine all SIFT descriptors for clustering
        all_descriptors = np.vstack([desc for desc in sift_descriptors if desc.size > 0])

        # Perform KMeans clustering to generate visual words
        self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
        self.kmeans.fit(all_descriptors)

        for descriptors in sift_descriptors:
            histogram = self.create_bovw_histogram(descriptors)
            image_histograms.append(histogram)

        # Scale the histograms
        image_histograms = self.scaler.fit_transform(image_histograms)
        return np.array(image_histograms), np.array(labels)

    def train(self, data, labels):
        """Train a Random Forest classifier on the dataset."""
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

    def save_model(self, model_path):
        """Save the trained model and related components."""
        joblib.dump((self.model, self.scaler, self.kmeans), model_path)

    def load_model(self, model_path):
        """Load a previously saved model and related components."""
        self.model, self.scaler, self.kmeans = joblib.load(model_path)

    def predict(self, image):
        """Predict the class of a single image."""
        descriptors = self.extract_sift_features(image)
        histogram = self.create_bovw_histogram(descriptors)
        histogram = self.scaler.transform([histogram])
        return self.model.predict(histogram)[0]

# Example usage
def main():
    dataset_path = "Classifier\Gesture Image Pre-Processed Data"
    model_path = "gesture_classifier_bovw.pkl"

    classifier = GestureClassifierWithBoVW(dataset_path)
    classifier.load_class_names(dataset_path)
    data, labels = classifier.load_dataset()
    classifier.train(data, labels)
    classifier.save_model(model_path)

if __name__ == "__main__":
    main()
