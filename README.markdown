# Image Captioning and Segmentation Project

This project combines **Image Captioning** and **Image Segmentation** using deep learning. It generates descriptive captions for images and segments objects within them, built with TensorFlow, Detectron2, and Streamlit.

## Features
- **Image Captioning**: Generates text descriptions for images using InceptionV3 (placeholder captioning).
- **Image Segmentation**: Identifies and labels objects using Detectron2's Mask R-CNN.
- **Streamlit App**: User-friendly interface for uploading and processing images.
- **Deployment**: Hosted on Streamlit Cloud.

## Setup and Deployment
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/<your-username>/image-caption-segmentation.git
   ```
2. **Test in Google Colab**:
   - Open [Google Colab](https://colab.research.google.com).
   - Run `install_dependencies.ipynb` and `download_sample.ipynb` to set up and test.
   - Test individual components (`captioning.py`, `segmentation.py`, `utils.py`).
3. **Deploy to Streamlit Cloud**:
   - Push the repository to GitHub.
   - Go to [Streamlit Cloud](https://streamlit.io/cloud).
   - Connect your GitHub repo and deploy `app.py`.
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