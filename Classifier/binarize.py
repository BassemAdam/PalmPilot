from PIL import Image
import numpy as np

# Load the image
image_path = 'image.png'
image = Image.open(image_path)

# Convert image to grayscale
gray_image = image.convert('L')

# Convert grayscale to binary mask using a threshold
threshold = 128  # This value can be adjusted
binary_mask = gray_image.point(lambda p: p > threshold and 255)

# Invert the binary mask by reversing black and white
inverted_mask = binary_mask.point(lambda p: 255 - p)

# Save the inverted binary mask
inverted_mask_path = 'inverted_hand_mask.png'
inverted_mask.save(inverted_mask_path)
inverted_mask_path

