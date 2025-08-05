import cv2
import numpy as np

def load_image(image_path):
    """Load and return image as numpy array."""
    return cv2.imread(image_path)

def display_results(image, masks, caption):
    """Prepare results for display (used in Streamlit)."""
    # For simplicity, return image and caption; masks need visualization logic
    return image, caption