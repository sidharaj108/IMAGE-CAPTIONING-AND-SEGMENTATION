import streamlit as st
import os
import cv2
from PIL import Image
from captioning import generate_caption
from segmentation import perform_segmentation

st.title("Image and Video Captioning with Segmentation")

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "png", "jpeg", "mp4", "mov"])

if uploaded_file is not None:
    # Determine file type
    file_type = uploaded_file.type.split('/')[0]  # 'image' or 'video'
    temp_file_path = f"temp_file.{uploaded_file.name.split('.')[-1]}"

    # Save uploaded file with error handling
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        st.success(f"File uploaded successfully: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        st.stop()

    if file_type == "image":
        # Display image
        st.image(temp_file_path, caption="Uploaded Image", use_column_width=True)

        # Generate caption
        caption = generate_caption(temp_file_path)
        st.write("Generated Caption:", caption)

        # Perform segmentation
        segmentation_result = perform_segmentation(temp_file_path)
        st.write("Segmentation Classes:", segmentation_result.get("classes", []))
        st.write("Segmentation Scores:", segmentation_result.get("scores", []))

    elif file_type == "video":
        # Process video
        st.video(temp_file_path)

        # Extract frames and generate captions
        try:
            vidcap = cv2.VideoCapture(temp_file_path)
            if not vidcap.isOpened():
                st.error("Error: Could not open video file.")
                st.stop()

            # Sample up to 5 frames
            frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = vidcap.get(cv2.CAP_PROP_FPS)
            sample_interval = max(1, int(frame_count / 5))
            frame_captions = []

            count = 0
            frame_idx = 0
            while vidcap.isOpened() and count < 5:
                vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                success, frame = vidcap.read()
                if not success:
                    break
                frame_path = f"temp_frame_{count}.jpg"
                cv2.imwrite(frame_path, frame)
                
                caption = generate_caption(frame_path)
                frame_captions.append((frame_idx / fps, caption))
                
                frame_image = Image.open(frame_path)
                st.image(frame_image, caption=f"Frame at {frame_idx / fps:.2f}s: {caption}", use_column_width=True)
                
                if os.path.exists(frame_path):
                    os.remove(frame_path)
                
                count += 1
                frame_idx += sample_interval

            vidcap.release()
            st.write("Video Frame Captions:")
            for time, caption in frame_captions:
                st.write(f"Time {time:.2f}s: {caption}")
        except Exception as e:
            st.error(f"Error processing video: {str(e)}")

    # Clean up temp file
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
