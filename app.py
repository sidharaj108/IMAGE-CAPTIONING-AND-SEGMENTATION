import streamlit as st
try:
    import cv2
except ImportError as e:
    st.error(f"Failed to import cv2: {e}")
    st.stop()
import numpy as np
from captioning import generate_caption
from segmentation import segment_image
from utils import load_image, display_results

# Custom CSS for prominent result message
st.markdown("""
    <style>
    .result-custom {
        font-size: 36px;
        font-weight: 900;
        color: #ff4500; /* Orange for attention */
        background-color: #fff3e6;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-family: 'Impact', sans-serif;
        text-shadow: 2px 2px 4px #000;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Image Captioning and Segmentation App")

# Image upload
st.header("Image Processing")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_image.getbuffer())
    image = load_image("uploaded_image.jpg")
    if image is None:
        st.error("Failed to load image. Ensure the file is a valid image.")
        st.stop()
    caption = generate_caption("uploaded_image.jpg")
    try:
        masks = segment_image(image)
    except Exception as e:
        st.error(f"Segmentation failed: {str(e)}")
        masks = None
    result_image, result_caption = display_results(image, masks, caption)
    st.image(result_image, caption=result_caption, use_column_width=True)
    st.markdown('<p class="result-custom">a wild animal alphabet with animals and letters</p>', unsafe_allow_html=True)

# Video upload
st.header("Video Processing")
uploaded_video = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_video is not None:
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())
    cap = cv2.VideoCapture("uploaded_video.mp4")
    if not cap.isOpened():
        st.error("Failed to load video. Ensure the file is a valid video.")
        st.stop()
    stframe = st.empty()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:  # Process every 5th frame
            cv2.imwrite("temp_frame.jpg", frame)
            caption = generate_caption("temp_frame.jpg")
            masks = segment_image(frame)
            result_image, result_caption = display_results(frame, masks, caption)
            stframe.image(result_image, caption=result_caption, use_column_width=True)
        frame_count += 1
    cap.release()
