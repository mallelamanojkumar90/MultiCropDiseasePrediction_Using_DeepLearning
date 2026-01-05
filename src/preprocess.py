import cv2
import numpy as np
from PIL import Image

def preprocess_image(image, target_size=(224, 224)):
    """
    Resize and normalize the image for model inference.
    
    Args:
        image: PIL Image or numpy array
        target_size: tuple (height, width)
        
    Returns:
        Processed image tensor ready for model input.
    """
    # Convert PIL image to numpy array if necessary
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize image - Area interpolation is best for shrinking high-res photos
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    # Convert to RGB if it's BGR (OpenCV default)
    if image.shape[-1] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype('float32') / 255.0
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image
