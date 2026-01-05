import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
import os
import json

# Define the classes based on common PlantVillage datasets
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

MODEL_MAP = {
    "Apple": "apple_model.h5",
    "Cherry": "cherry_model.h5",
    "Corn": "corn_model.h5",
    "Grape": "grape_model.h5",
    "Peach": "peach_model.h5",
    "Pepper": "pepper_model.h5",
    "Potato": "potato_model.h5",
    "Strawberry": "strawberry_model.h5",
    "Tomato": "tomato_model.h5"
}

def get_crop_classes(crop_name):
    """Returns the list of disease classes for a specific crop, ensuring 'healthy' is index 0."""
    # Normalize crop name for matching
    search_name = crop_name.split(' ')[0].lower() if ' ' in crop_name else crop_name.lower()
    
    # Filter classes for this crop
    crop_specific = [c.split('___')[1] for c in CLASS_NAMES if c.lower().startswith(search_name)]
    
    # Special handling for "Pepper"
    if not crop_specific and search_name == "pepper":
         crop_specific = [c.split('___')[1] for c in CLASS_NAMES if "pepper" in c.lower()]
    
    # Move 'healthy' to the front (index 0) if it exists
    # This aligns with the provided pre-trained models where 0=Healthy
    healthy_class = None
    other_classes = []
    
    for c in sorted(crop_specific):
        if 'healthy' in c.lower():
            healthy_class = c
        else:
            other_classes.append(c)
            
    if healthy_class:
        return [healthy_class] + other_classes
    return other_classes

def get_class_names():
    """Returns the class names from the saved json if it exists, else returns defaults."""
    json_path = os.path.join('models', 'class_indices.json')
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            class_indices = json.load(f)
        # Sort by index value
        return [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
    return CLASS_NAMES

def build_model(num_classes=len(CLASS_NAMES)):
    """
    Build a ResNet50 based model for transfer learning.
    Using Sequential wrapper to match hierarchical weight saving (often 3-4 top-level layers).
    """
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 

    model = tf.keras.Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_trained_model(crop_name):
    """
    Loads a specific crop model. Tries loading as a full model first,
    then falls back to building the architecture and loading weights.
    """
    model_file = MODEL_MAP.get(crop_name)
    if not model_file:
        return f"Unknown crop: {crop_name}"
        
    model_path = os.path.join('models', model_file)
    
    if not os.path.exists(model_path):
        return f"Model file not found: {model_path}"

    classes = get_crop_classes(crop_name)
    if not classes:
        return f"No classes found for {crop_name}"

    # Legacy Naming Fix: Keras 3 doesn't like '/' in names. 
    # We try to load weights by skipping name mismatch if necessary.
    
    # Priority 1: Try building architecture and loading weights (Smarter for Keras 3/2 mismatch)
    try:
        model = build_model(num_classes=len(classes))
        # by_name=False handles positional loading which works for 
        # Sequential models with matching layer counts
        model.load_weights(model_path)
        print(f"Weights for {model_file} loaded successfully via position.")
        return model
    except Exception as e_pos:
        print(f"Positional load failed: {e_pos}. Trying by_name=True...")
        try:
            model = build_model(num_classes=len(classes))
            model.load_weights(model_path, by_name=True, skip_mismatch=True)
            print(f"Weights for {model_file} loaded successfully via name matching.")
            return model
        except Exception as e_name:
            print(f"Name-based load failed: {e_name}. Trying full model load...")
            # Priority 2: Try loading the full model
            try:
                # We use compile=False to avoid issues with older optimizers
                model = tf.keras.models.load_model(model_path, compile=False)
                print(f"Full model {model_file} loaded successfully.")
                return model
            except Exception as e_full:
                error_details = f"Pos Error: {str(e_pos)} | Name Error: {str(e_name)} | Full Load Error: {str(e_full)}"
                print(error_details)
                return error_details

def predict(model, processed_image, crop_name):
    """
    Perform inference on the processed image for a specific crop.
    """
    classes = get_crop_classes(crop_name)
    predictions = model.predict(processed_image)
    class_idx = predictions.argmax()
    confidence = predictions[0][class_idx]
    
    disease = classes[class_idx].replace('_', ' ')
    
    return crop_name, disease, confidence
