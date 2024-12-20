import cv2
import numpy as np
import os

# Function to generate labels for images
def generate_labels(image_directory):
    labels = []
    image_paths = []
    class_to_label = {'a': 0, 'b': 1, 'c': 2, 'd': 3}  # Map classes 'a', 'b', 'c', 'd' to 0, 1, 2, 3
    
    # Iterate over the subdirectories (one for each letter/class)
    for class_name, label in class_to_label.items():
        class_dir = os.path.join(image_directory, class_name)
        
        if os.path.isdir(class_dir):
            # Iterate through images in each class directory
            for filename in os.listdir(class_dir):
                if filename.endswith(".png") or filename.endswith(".jpg"):  # Assuming images are in PNG or JPG format
                    image_paths.append(os.path.join(class_dir, filename))
                    labels.append(label)

    return np.array(image_paths), np.array(labels)

# Example usage: Replace with the actual dataset path
image_directory = "path_to_your_data/Gesture Image Pre-Processed"  # Root directory for the dataset
image_paths, labels = generate_labels(image_directory)

print(f"Generated {len(labels)} labels")
