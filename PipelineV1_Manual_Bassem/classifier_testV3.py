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
        self.class_names = ['devil sign', 'fist ', 'palm']  # Replace with actual class names
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
    # Read the test image
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded successfully
    classifier = GestureClassifierWithSIFT()

    # Predict the gesture and its probability
    gesture, probability = classifier.predict(image)

    # Print the result
    x = probability.max()
    return gesture

import cv2
import numpy as np

def create_hand_mask_expanded(image):
    """Create a binary mask using YCrCb and LAB color spaces."""
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)

    # Define thresholds for skin detection
    lower_ycrcb = np.array([0, 140, 100], dtype=np.uint8)
    upper_ycrcb = np.array([255, 180, 140], dtype=np.uint8)
    lower_lab = np.array([20, 135, 125], dtype=np.uint8)
    upper_lab = np.array([255, 175, 145], dtype=np.uint8)

    # Create masks for YCrCb and LAB
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)
    mask_lab = cv2.inRange(lab, lower_lab, upper_lab)
    combined_mask = cv2.bitwise_or(mask_ycrcb, mask_lab)

    # Clean the mask with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( 10,10))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    #combined_mask = cv2.dilate(combined_mask, kernel, iterations=3)
    return combined_mask.astype(np.uint8)

def find_hand_contours(skin_mask, image):
    """Find the largest contour and compute its convex hull."""
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
    if not contours:
        return image, None, None

    # Sort contours by area
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    hand_contour = contours[0]
    # Visualize the convex hull
    output_image = image.copy()
    try:
        hull = cv2.convexHull(hand_contour)
        cv2.drawContours(output_image, [hull], -1, (255, 0, 0), 2)
    except Exception as e:
        print(f"Error processing contour: {e}")
        return output_image, None, None

    return output_image, hand_contour, hull

def create_color_mask(new_screen, hull, square_size=64):
    """
    Create a mask based on LAB color similarity within a bounding box using adaptive thresholding.
    """
    # Find the bounding box of the hull
    x, y, w, h = cv2.boundingRect(hull)

    # Calculate the center and create an inner box
    center_x, center_y = x + w // 2, y + h // 2
    inner_x, inner_y = center_x - square_size // 2, center_y - square_size // 2
    inner_x, inner_y = max(0, inner_x), max(0, inner_y)
    inner_w, inner_h = square_size, square_size


    # Convert to LAB color space
    lab_screen = cv2.cvtColor(new_screen, cv2.COLOR_BGR2LAB)

    # Extract inner box region and compute mean LAB color
    inner_box_region = lab_screen[inner_y:inner_y + inner_h, inner_x:inner_x + inner_w]
    avg_lab_color = cv2.mean(inner_box_region)[:3]

    # Apply LAB color thresholding to the bounding box
    bounding_box_region = lab_screen[y:y + h, x:x + w]
    diff = cv2.absdiff(bounding_box_region, np.uint8(avg_lab_color))
    diff_norm = np.linalg.norm(diff, axis=2).astype(np.uint8)

    # Apply adaptive thresholding to create a binary mask
    binary_mask = cv2.adaptiveThreshold(diff_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    
    lower_lab = np.array([20, 135, 125], dtype=np.uint8)
    upper_lab = np.array([255, 175, 145], dtype=np.uint8)
    mask_lab = cv2.inRange(bounding_box_region, lower_lab, upper_lab)
    final_mask = cv2.bitwise_and(binary_mask, mask_lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ( 10,10))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)
    final_mask=resize_with_aspect_ratio(final_mask, target_size=(200, 200), pad_color=0)
    return final_mask

def resize_with_aspect_ratio(image, target_size=(200, 200), pad_color=0):
    h, w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    if w == 0 or h == 0:
        print("Invalid bounding box dimensions:", w, h)
        return None
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w), pad_color, dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_image
    return canvas


def process_frame(frame):
    """Process the input frame to detect the hand region."""
    skin_mask = create_hand_mask_expanded(frame)
    output_image, hand_contour, hull = find_hand_contours(skin_mask, frame)

    
    
    cx = 0
    cy = 0
    if hull is not None:
        hull_points = np.squeeze(hull)
        cx = np.mean(hull_points[:, 0]).astype(int)  # Average x-coordinates
        cy = np.mean(hull_points[:, 1]).astype(int)  # Average y-coordinates
        hand_region = create_color_mask(frame, hull)
    else:
        hand_region = np.zeros_like(frame)

    return skin_mask, output_image, hand_region, [cx, cy]

def detect_movement(hand_region, prev_hand_region,previous_center,center,threshold=50):
    """Detect movement between the current and previous hand regions."""
    if hand_region is None or prev_hand_region is None:
        return False
    print(previous_center)
    print(center)
    print(hand_region)
    print(prev_hand_region)
    if (hand_region == prev_hand_region) and (hand_region== "Palm"):
        if previous_center[1] - center[1] > threshold:
            return "Scroll up"
        elif abs (previous_center[1] - center[1]) > threshold:
            return "Scoll down"
        return False
    if (hand_region == prev_hand_region) and (hand_region== "Devil Sign"):
        if previous_center[0] - center[0] > threshold:
            return "zoom in"
        elif abs (previous_center[0] - center[0]) > threshold:
            return "zoom out"
        return False


def main():
    previous_center = [0,0]
    previous_decision = None
    new_decision = None
    new_center = [0,0]
    cap = cv2.VideoCapture(0)
    frames = 0 ;
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        frame = cv2.flip(frame, 1)

        # Display results
        cv2.imshow("Original", frame)
        skin_mask, output_image, hand_region,center = process_frame(frame)
        cv2.imshow("Skin Mask", skin_mask)
        cv2.imshow("Output", output_image)
        cv2.imshow("Hand Region", hand_region)
        if frames % 10 == 0:
            new_decision = test_single_image(hand_region)
            new_center = center
            movement = detect_movement(new_decision, previous_decision,previous_center,new_center)
            if movement:
                print(movement)
            previous_decision = new_decision
            previous_center = new_center
            #print(center)
            frames = 0


        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




