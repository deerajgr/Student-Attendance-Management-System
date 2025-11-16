import cv2
import numpy as np
from deepface import DeepFace
import pickle
import os

ENCODINGS_FILE = "encodings.pkl"

def encode_face(image_path_or_pil, student_id):
    """
    Encode a face from an image path or PIL image.
    Returns the encoding or None if no face/multiple faces.
    """
    try:
        if isinstance(image_path_or_pil, str):
            image = cv2.imread(image_path_or_pil)
        else:
            image = cv2.cvtColor(np.array(image_path_or_pil), cv2.COLOR_RGB2BGR)
        
        # Use DeepFace for embedding
        embedding = DeepFace.represent(image, model_name='Facenet', enforce_detection=False)[0]['embedding']
        return np.array(embedding), None
    except Exception as e:
        return None, f"Error encoding face: {str(e)}"

def save_encodings(encodings_dict):
    """Save encodings dict to pickle file."""
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(encodings_dict, f)

def load_encodings():
    """Load encodings dict from pickle file."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            return pickle.load(f)
    return {}