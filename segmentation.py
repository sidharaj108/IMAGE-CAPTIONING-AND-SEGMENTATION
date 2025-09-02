from ultralytics import YOLO
import cv2
import streamlit as st

try:
    model = YOLO("yolov8m-seg.pt")  # Automatically downloads if not cached
except Exception as e:
    st.error(f"Failed to load YOLOv8 model: {str(e)}")
    raise

def segment_image(image):
    try:
        results = model(image, verbose=False)
        return results[0].masks.data.cpu().numpy() if results[0].masks is not None else None
    except Exception as e:
        st.error(f"Segmentation failed: {str(e)}")
        return None
