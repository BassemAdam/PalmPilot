import cv2
import numpy as np
from joblib import dump, load
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from generate_mask_labels import generate_labels

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

# Function to load images, extract features, and train the classifier
def train_classifier(image_paths, labels):
    features_list = []

    # Load each image, extract features, and append to feature list
    for path in image_paths:
        # Load the image as a binary mask (assuming image is already in binary format)
        binary_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is not None:
            # Resize image if necessary (e.g., 128x128)
            resized_mask = cv2.resize(binary_mask, (128, 128))  # Resize to the expected size
            _, binary_resized_mask = cv2.threshold(resized_mask, 127, 255, cv2.THRESH_BINARY)
            features = extract_features(binary_resized_mask)
            features_list.append(features)

    # Convert list to numpy array
    X = np.array(features_list)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Initialize and train the classifier (SVM)
    classifier = SVC(kernel='linear', C=1)
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Save the trained classifier
    dump(classifier, "gesture_svm_model_4_classes.joblib")
    print("Model saved to 'gesture_svm_model_4_classes.joblib'")
image_paths, labels = generate_labels("./Gesture Image Pre-Processed Data")    

# Example usage: Train classifier on the filtered dataset (4 classes: 'a', 'b', 'c', 'd')
train_classifier(image_paths, labels)
