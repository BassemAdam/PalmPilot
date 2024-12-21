import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage import exposure

# Load the saved model and scaler
#model = joblib.load("gesture_classifier_model.pkl")
#scaler = joblib.load("gesture_classifier_scaler.pkl")

class GestureClassifierWithSIFT:
    def __init__(self):
        self.class_names = ['A', 'B', 'C', 'D']  # Replace with actual class names
        self.class_names = []
        self.model = None
        self.sift = cv2.SIFT_create()
        self.kmeans = None
        self.visual_words = None
        self.num_clusters = 50
    
    def extract_sift_features(self, image):
        """Extract SIFT descriptors from an image."""
        # If the image is already grayscale, skip conversion
        if len(image.shape) == 2:
            gray = image
        elif len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Invalid image format. Expected 2D (grayscale) or 3D (BGR).")

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
    
    def load_model(self, model_path):
        """Load a previously saved model and related components."""
        self.model, self.scaler, self.kmeans = joblib.load(model_path)
        return 


    def predict(self, image):
        """Predict the class of a single image and include probability."""
        self.load_model("gesture_classifier_bovw.pkl")
        descriptors = self.extract_sift_features(image)
        histogram = self.create_bovw_histogram(descriptors)
        histogram = self.scaler.transform([histogram])
        
        probabilities = self.model.predict_proba(histogram)[0]
        predicted_class = self.model.predict(histogram)[0]  # No need for int()
        predicted_class_name = predicted_class  # Directly use the predicted class

        return predicted_class_name, probabilities



# Test function
def test_single_image(image):
    classifier = GestureClassifierWithSIFT()
    predicted_class, probabilities = classifier.predict(image)
    
    print(f"Predicted Class: {predicted_class}")
    print("Probabilities:")
    for class_name, prob in zip(classifier.class_names, probabilities):
        print(f"  {class_name}: {prob:.2f}")




def process_segments(image):
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



# Run the test after the model is trained
if __name__ == "__main__":
    # Ensure that the model and scaler are trained first
    # You can call main() function here if not already done
    # main()  # Uncomment if model needs to be trained
    cap = cv2.VideoCapture(0)
    frames = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        original, segmented, blue_part, red_part = process_segments(frame)
        
        if original is not None:
            cv2.imshow("Original Image", cv2.resize(original, (600, 400)))
        if segmented is not None:
            cv2.imshow("Red and Blue Screen", cv2.resize(segmented, (600, 400)))
        if blue_part is not None:
            #cv2.imshow("Blue Part", blue_part)
            #test_single_image(blue_part)
            if frames % 10 == 0:
                test_single_image(red_part)
                
        if red_part is not None:
            #cv2.imshow("Red Part", red_part)
            if frames % 10 == 0:
                test_single_image(red_part)
                frames = 0

        frames = frames + 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()  # Provide a test image path


