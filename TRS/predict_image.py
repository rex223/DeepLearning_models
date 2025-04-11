import numpy as np
import tensorflow as tf
# Index mapping for your model
train_indices_class = {
    '0': 0, '1': 1, '10': 2, '11': 3, '12': 4, '13': 5, '14': 6, '15': 7, '16': 8, '17': 9, 
    '18': 10, '19': 11, '2': 12, '20': 13, '21': 14, '22': 15, '23': 16, '24': 17, '25': 18, 
    '26': 19, '27': 20, '28': 21, '29': 22, '3': 23, '30': 24, '31': 25, '32': 26, '33': 27, 
    '34': 28, '35': 29, '36': 30, '37': 31, '38': 32, '39': 33, '4': 34, '40': 35, '41': 36, 
    '42': 37, '5': 38, '6': 39, '7': 40, '8': 41, '9': 42
}

# Reverse mapping to get class index from prediction index
reverse_train_indices = {v: k for k, v in train_indices_class.items()}

# Sign class information
SIGN_CLASSES = {
    0: "Speed limit (20km/h)",
    1: "Speed limit (30km/h)",
    2: "Speed limit (50km/h)",
    3: "Speed limit (60km/h)",
    4: "Speed limit (70km/h)",
    5: "Speed limit (80km/h)",
    6: "End of speed limit (80km/h)",
    7: "Speed limit (100km/h)",
    8: "Speed limit (120km/h)",
    9: "No passing",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Right-of-way at the next intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "Vehicles over 3.5 metric tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve to the left",
    20: "Dangerous curve to the right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles crossing",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End of all speed and passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Roundabout mandatory",
    41: "End of no passing",
    42: "End of no passing by vehicles over 3.5 metric tons"
}

# Generic warning messages for each sign type
WARNING_MESSAGES = {
    0: "Observe 20km/h speed limit ahead",
    1: "Observe 30km/h speed limit ahead",
    2: "Observe 50km/h speed limit ahead",
    3: "Observe 60km/h speed limit ahead",
    4: "Observe 70km/h speed limit ahead",
    5: "Observe 80km/h speed limit ahead",
    6: "End of 80km/h speed limit, normal rules apply",
    7: "Observe 100km/h speed limit ahead",
    8: "Observe 120km/h speed limit ahead",
    9: "No passing zone ahead",
    10: "No passing for vehicles over 3.5 metric tons",
    11: "Yield to traffic at the next intersection",
    12: "You have priority at upcoming junctions",
    13: "Yield to crossing traffic",
    14: "Come to a complete stop",
    15: "No vehicles allowed in this area",
    16: "No vehicles over 3.5 metric tons allowed",
    17: "Do not enter this road",
    18: "Caution! Hazard ahead",
    19: "Dangerous left curve ahead",
    20: "Dangerous right curve ahead",
    21: "Series of dangerous curves ahead",
    22: "Bumpy road ahead, reduce speed",
    23: "Slippery road surface ahead, drive carefully",
    24: "Road narrows on the right side",
    25: "Road work ahead, drive carefully",
    26: "Traffic signals ahead",
    27: "Pedestrian crossing ahead",
    28: "Children crossing area ahead",
    29: "Bicycles crossing ahead",
    30: "Beware of ice or snow on road",
    31: "Wild animals may cross the road",
    32: "End of all speed and passing restrictions",
    33: "Turn right at the next intersection",
    34: "Turn left at the next intersection",
    35: "Proceed straight ahead only",
    36: "Go straight or turn right only",
    37: "Go straight or turn left only",
    38: "Keep to the right of divider",
    39: "Keep to the left of divider",
    40: "Roundabout ahead, drive clockwise",
    41: "End of no passing zone",
    42: "End of no passing restriction for vehicles over 3.5 metric tons"
}

def load_model():
    """
    Load the trained model from the saved file
    """
    try:
        model = tf.keras.models.load_model('final_model.h5')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_image(img, model=None, confidence_threshold=0.5):
    """
    Predict the class of a traffic sign image
    
    Args:
        img: Preprocessed image ready for prediction
        model: Loaded TensorFlow model (will load if None)
        confidence_threshold: Minimum confidence for reliable prediction
    
    Returns:
        Tuple of (class_id, confidence)
    """
    if img is None:
        print("❌ Error: Image not found!")
        return -1, 0.0
    
    # If model isn't provided, load it
    if model is None:
        model = load_model()
        if model is None:
            return -1, 0.0
    
    # Ensure image is in correct shape with batch dimension
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    try:
        # Predict
        pred = model.predict(img, verbose=0)
        confidence = np.max(pred)
        pred_index = int(np.argmax(pred))
        
        # Map prediction index to real class label using reverse mapping
        pred_class_str = reverse_train_indices.get(pred_index)
        if pred_class_str is None:
            print(f"⚠️ Could not map prediction index {pred_index} to a class")
            return -1, confidence
            
        pred_class = int(pred_class_str)
        
        # Print information for debugging
        if confidence < confidence_threshold:
            print("⚠️ Not sure about the outcome! Use your judgment")
        
        sign_name = SIGN_CLASSES.get(pred_class, "Unknown")
        warning = WARNING_MESSAGES.get(pred_class, "No warning information")
        
        print(f"🚧 Sign Detected: {sign_name}")
        print(f"⚠️ Warning: {warning}")
        print(f"✅ Confidence: {confidence * 100:.2f}%")
        
        return pred_class, confidence
    
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return -1, 0.0

def get_sign_info(class_id):
    """
    Get sign name and warning message for a given class ID
    
    Args:
        class_id: The class ID to look up
        
    Returns:
        Tuple of (sign_name, warning_message)
    """
    sign_name = SIGN_CLASSES.get(class_id, "Unknown")
    warning = WARNING_MESSAGES.get(class_id, "No warning information")
    return sign_name, warning

# # For standalone testing
# if __name__ == "__main__":
#     import sys
    
#     # Example usage
#     if len(sys.argv) > 1:
#         from PIL import Image
#         import numpy as np
        
#         try:
#             # Import the exact function, not the whole module
#             import preprocess_image
            
#             # Load and preprocess image
#             img_path = sys.argv[1]
#             image = Image.open(img_path)
#             img_array = np.array(image)
            
#             # Make sure we're calling preprocess_image as a function, not a module
#             processed_img = preprocess_image(img_array)
            
#             # Predict
#             class_id, confidence = predict_image(processed_img)
            
#             # Print result
#             if class_id >= 0:
#                 sign_name, warning = get_sign_info(class_id)
#                 print(f"\nPrediction: {sign_name}")
#                 print(f"Warning: {warning}")
#                 print(f"Confidence: {confidence * 100:.2f}%")
#         except ImportError:
#             print("❌ Error: Could not import preprocess_image function. Check your image_preprocessing.py file.")
#         except Exception as e:
#             print(f"❌ Error: {e}")
#     else:
#         print("Usage: python predict_image.py <image_path>")