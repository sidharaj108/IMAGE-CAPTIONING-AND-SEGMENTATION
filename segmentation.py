from ultralytics import YOLO
import cv2
import os

# Path to store the model
model_path = "yolov8m-seg.pt"

# Download model if not present
if not os.path.exists(model_path):
    from ultralytics.hub import download
    download("yolov8m-seg.pt", model_path)

# Load YOLOv8 model
model = YOLO(model_path)

def segment_image(image):
    results = model(image, verbose=False)
    return results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
