import cv2
import numpy as np
import random

def load_image(image_path):
    return cv2.imread(image_path)

def display_results(image, masks, caption):
    if masks is None:
        return image, caption
    
    # Convert image to RGB for consistency
    result_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Generate random colors for each mask
    num_masks = masks.shape[0] if len(masks.shape) > 2 else 1
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_masks)]
    
    # Overlay masks on the image
    for i in range(num_masks):
        mask = masks[i] if len(masks.shape) > 2 else masks
        mask = (mask > 0).astype(np.uint8) * 255  # Convert to binary mask
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create colored mask
        colored_mask = np.zeros_like(image)
        colored_mask[mask == 255] = colors[i]
        
        # Blend mask with image (50% transparency)
        alpha = 0.5
        result_image = cv2.addWeighted(result_image, 1.0, colored_mask, alpha, 0.0)
    
    return result_image, caption
