import streamlit as st
import cv2
import numpy as np
from captioning import extract_features, generate_caption
from segmentation import segment_image
from utils import load_image, display_results

st.title("Image Captioning and Segmentation App")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # Save uploaded file
    with open("uploaded_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Process image
    image = load_image("uploaded_image.jpg")
    features = extract_features("uploaded_image.jpg")
    caption = generate_caption(features)
    masks = segment_image(image)
    
    # Display results
    result_image, result_caption = display_results(image, masks, caption)
    st.image(result_image, caption=result_caption, use_column_width=True)
    st.write("Segmentation masks generated (visualization requires additional processing)")