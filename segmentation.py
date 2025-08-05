from ultralytics import YOLO
import cv2

model = YOLO("yolov8m-seg.pt")

def segment_image(image):
    results = model(image, verbose=False)
    return results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
