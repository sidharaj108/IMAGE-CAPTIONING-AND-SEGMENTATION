import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

cache_dir = os.environ.get('TRANSFORMERS_CACHE', '/home/user/.cache/huggingface/transformers')
processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        try:
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=cache_dir)
            model = BlipForConditionalGeneration.from_pretrained(
                "Salesforce/blip-image-captioning-base",
                cache_dir=cache_dir,
                torch_dtype=torch.float32
            )
            print("BLIP model loaded successfully.")
        except Exception as e:
            print(f"Error loading BLIP model: {e}")
    return processor, model

def generate_caption(image_path):
    load_model()
    if processor is None or model is None:
        return "Error: BLIP model not loaded."
    try:
        image = Image.open(image_path).convert('RGB')
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs, max_length=50, num_beams=5)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error generating caption: {str(e)}"

load_model()
