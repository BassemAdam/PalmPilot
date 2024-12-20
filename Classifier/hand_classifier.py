import cv2
import numpy as np
from joblib import dump
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from generate_mask_labels import generate_labels

# Function to extract features directly from the raw image
def extract_features_from_image(image):
    features = []
    
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Thresholding to separate the object from the background
    _, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    # Contour detection
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        perimeter = cv2.arcLength(largest_contour, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = w / h
        
        # Fit an ellipse and calculate eccentricity
        if len(largest_contour) >= 5:
            (x_center, y_center), (major_axis, minor_axis), angle = cv2.fitEllipse(largest_contour)
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis != 0 else 0
        else:
            eccentricity = 0
        
        features.extend([area, perimeter, circularity, aspect_ratio, eccentricity])
    else:
        features.extend([0, 0, 0, 0, 0])

    # Convex hull features
    if len(contours) > 0:
        hull = cv2.convexHull(largest_contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        features.append(solidity)
    else:
        features.append(0)

    # Hu Moments
    moments = cv2.moments(thresholded)
    hu_moments = cv2.HuMoments(moments).flatten()
    features.extend(hu_moments)

    return np.array(features)

# Function to train the classifier
def train_classifier(image_paths, labels):
    features_list = []

    for path in image_paths:
        image = cv2.imread(path)
        if image is not None:
            features = extract_features_from_image(image)
            features_list.append(features)

    # Convert list to numpy array
    X = np.array(features_list)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    # Initialize and train the classifier
    classifier = SVC(kernel='rbf', C=10, gamma='scale')  # Use RBF kernel for non-linear separation
    classifier.fit(X_train, y_train)

    # Evaluate the classifier
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classification Accuracy: {accuracy:.2f}")

    # Save the trained classifier
    dump(classifier, "gesture_svm_model_no_mask.joblib")
    print("Model saved to 'gesture_svm_model_no_mask.joblib'")

# Main program
if __name__ == "__main__":
    # Load image paths and labels using a helper function
    image_paths, labels = generate_labels("./Gesture Image Pre-Processed Data")
    
    # Train the classifier
    train_classifier(image_paths, labels)
