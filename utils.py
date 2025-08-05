import cv2
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path)

def display_results(image, masks, caption):
    return image, caption
