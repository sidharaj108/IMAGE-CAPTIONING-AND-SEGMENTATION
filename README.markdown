# Image Captioning and Segmentation Project

This project combines **Image And Video Captioning** and **Image Segmentation** using deep learning. It generates descriptive captions for images and videos and segments objects within them, built with TensorFlow, Detectron2, BLIP and Streamlit.

## Features
- **Image Captioning**: Generates text descriptions for images using InceptionV3 (placeholder captioning).
- **Image Segmentation**: Identifies and labels objects using Detectron2's Mask R-CNN.
- **Video Captioning**: Generates text description for video using BLIP.
- **Streamlit App**: User-friendly interface for uploading and processing images.
- **Deployment**: Hugging Face Cloud for proper utilization of Space.

## HERE IS MY APP :
https://huggingface.co/spaces/Sidharaj-Manek07/IMAGE-CAPTIONING-AND-SEGMENTATION/tree/main
## Setup and Deployment
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/image-caption-segmentation.git
   ```
2. **Test in Google Colab**:
   - Open [Google Colab](https://colab.research.google.com).
   - Run `install_dependencies.ipynb` and `download_sample.ipynb` to set up and test.
   - Test individual components (`captioning.py`, `segmentation.py`, `utils.py`).
3. **Deploy to Hugging Face**:
   - Clone the repository from Github to Hugging Face.
   - Go to [Hugging Face](https://huggingface.co/).
   - Hugging face will automatically running the `app.py` in 10-15 minutes as building the model.
   - Ensure `requirements.txt` is correct.

## File Structure
- `app.py`: Main Streamlit app.
- `captioning.py`: Image captioning logic.
- `segmentation.py`: Image segmentation logic.
- `utils.py`: Shared utilities.
- `requirements.txt`: Dependencies.
- `sample_image.jpg`: Sample image for testing.
- `README.md`: This file.

## Notes
- Captioning uses a placeholder; extend with BLIP or LSTM for production.
- Segmentation masks require additional visualization logic for full display.
- Video streaming is limited in Streamlit Cloud (upload-based only).

## License
MIT License
