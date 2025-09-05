import streamlit as st
from captioning import generate_caption
from segmentation import perform_segmentation
from utils import load_image  # Assuming this exists

st.title("Image Captioning and Segmentation")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])
if uploaded_file:
    image = load_image(uploaded_file)
    st.image(image, caption="Uploaded Image")
    caption = generate_caption(image)
    st.write(f"Caption: {caption}")
    segmented = perform_segmentation(image)
    st.image(segmented, caption="Segmented Image")
