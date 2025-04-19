# Environment variables first
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['CUDA_VISIBLE_DEVICES'] = '' 
os.environ['YOLO_VERBOSE'] = 'False'

# Set up async event loop
# import asyncio
# try:
#     asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
#     loop = asyncio.new_event_loop()
#     asyncio.set_event_loop(loop)
# except Exception:
#     pass

# import nest_asyncio
# nest_asyncio.apply()

# PyTorch import with patches
import torch
torch.classes_path = None
torch._C._get_tracing_state = lambda: None  # Prevents the error related to tracing state

# Standard libraries first
import time
import io
import numpy as np
import streamlit as st
import tensorflow as tf

# PyTorch import with proper error handling
import torch
torch.classes_path = None  # Prevent __path__ attribute access that's causing the error

# Image processing libraries
from PIL import Image
# Import OpenCV
import cv2
    
# from preprocess_image import preprocess_image
# from predict_image import *
from ultralytics import YOLO
# Set page configuration and title
st.set_page_config(page_title="Traffic Sign Recognition", layout="wide", initial_sidebar_state="expanded")

# Apply custom CSS for dark mode and styling
st.markdown("""
<style>
    /* Dark mode colors */
    :root {
        --background-color: #121212;
        --secondary-bg-color: #1e1e1e;
        --text-color: #f0f0f0;
        --accent-color: #00c6ff;
        --success-color: #4caf50;
        --warning-color: #ff9800;
    }

    /* Main background */
    .stApp {
        background-color: #34495E;
        color: var(--text-color);
    }

    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: var(--accent-color) !important;
        font-weight: bold !important;
    }

    /* Button styling */
    .stButton>button {
        background-color: var(--secondary-bg-color);
        color: var(--accent-color);
        border: 2px solid var(--accent-color);
        border-radius: 6px;
        padding: 12px 20px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: var(--accent-color);
        color: var(--background-color);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 198, 255, 0.3);
    }

    /* Card-like containers */
    .css-card {
        background-color: var(--secondary-bg-color);
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid var(--accent-color);
    }
    /* Fix for white border */
    .main .block-container {
        border-top: none !important;
        padding-top: 0 !important;
    }

    header[data-testid="stHeader"] {
        background: 
            /* Blue gradient at the end (adjust 30% to control transition point) */
            linear-gradient(90deg, transparent 0%, #34495E 36%, #00c6ff 100%),
            /* First road sign image (left-aligned) */
            url('https://www.shutterstock.com/image-photo/road-signs-banner-traffic-laws-260nw-2344101155.jpg') left 0% center/contain no-repeat,
            /* Second road sign image (centered) */
            right 10% center/contain no-repeat;
        
            background-blend-mode: overlay;
            height: 10vh !important;
            min-height: 5vh !important;
            padding: 3rem 2rem !important;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* Success message styling */
    .success-box {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid var(--success-color);
        padding: 10px 15px;
        border-radius: 5px;
        color: var(--success-color);
        margin: 10px 0;
    }

    /* File uploader */
    .uploadedFileData {
        background-color: var(--secondary-bg-color) !important;
        border-radius: 6px !important;
        padding: 10px !important;
    }
    /* Tab styling for better visibility in both dark and light modes */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
        background-color: var(--secondary-bg-color);
        border-radius: 8px;
        padding: 10px 20px;
        margin-bottom: 20px;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto;
        white-space: nowrap;
        font-weight: bold;
        color: #ffffff !important;
        background-color: rgba(0, 198, 255, 0.1);
        border-radius: 6px;
        padding: 8px 16px;
        border: 1px solid rgba(0, 198, 255, 0.3);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0, 198, 255, 0.2);
        border-color: var(--accent-color);
    }

    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 198, 255, 0.3) !important;
        border-color: var(--accent-color) !important;
        color: var(--accent-color) !important;
    }

    /* Make emoji icons more visible */
    .stTabs [data-baseweb="tab"] p {
        font-size: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
    }

    /* Add glow effect to tab labels for better contrast */
    .stTabs [data-baseweb="tab"] p {
        text-shadow: 0 0 5px rgba(0, 198, 255, 0.5);
    }
    /* Prediction result styling */
    .prediction-result {
        font-size: 24px;
        padding: 15px;
        margin: 15px 0;
        border-radius: 8px;
        text-align: center;
        background: linear-gradient(145deg, #1e1e1e, #2a2a2a);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }

    /* Image display enhancement */
    .stImage img {
        border-radius: 6px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }

    /* Progress bar colors */
    .stProgress > div > div {
        background-color: var(--accent-color) !important;
    }

    /* Container with gradient border */
    .gradient-border {
        position: relative;
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        background: var(--secondary-bg-color);
    }
    .gradient-border:before {
        content: '';
        position: absolute;
        top: -2px;
        right: -2px;
        bottom: -2px;
        left: -2px;
        background: linear-gradient(45deg, var(--accent-color), #ff7f50);
        border-radius: 12px;
        z-index: -1;
    }
    /* Camera input button styling */
    button[data-testid="stCameraInputButton"] {
        color: green !important;
        font-weight: bold !important;
        border-color: black !important;
    }

    button[data-testid="stCameraInputButton"]:hover {
        color: #ff4d4d !important;
        background-color: rgba(255, 0, 0, 0.1) !important;
    }

    button[data-testid="stCameraInputButton"]:active {
        color: #4caf50 !important;
        border-color: #4caf50 !important;
        background-color: rgba(76, 175, 80, 0.1) !important;
    }

    /* Ensure text inside the button is also properly colored */
    button[data-testid="stCameraInputButton"] p,
    button[data-testid="stCameraInputButton"] span {
        color: red !important;
    }

    button[data-testid="stCameraInputButton"]:active p,
    button[data-testid="stCameraInputButton"]:active span {
        color: #4caf50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Loading the models
def load_models():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "final_model.h5")
    yolo_model_path = os.path.join(base_dir, "best.pt")
    
    models = {}
    try:
        models['cnn'] = tf.keras.models.load_model(model_path)
        models['yolo'] = YOLO(yolo_model_path)
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None
def preprocess_image(img):
    # Resize, normalize and expand dims
    IMG_SIZE=32
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    # img=cv2.imread(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    return img
def display_image(img):
    st.image(img, caption="Input Image", use_column_width=True)
    st.write("Image displayed successfully.")
    
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

# def load_model():
#     """
#     Load the trained model from the saved file
#     """
#     try:
#         model = tf.keras.models.load_model('final_model.h5')
#         return model
#     except Exception as e:
#         print(f"Error loading model: {e}")
#         return None

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

def display_prediction_results(prediction_results):
    """
    Display prediction results from YOLO + CNN pipeline (camera path)
    prediction_results: List of dictionaries with prediction information
    """
    if not prediction_results:
        st.warning("No predictions to display.")
        return

    for i, result in enumerate(prediction_results):
        class_id = result["class_id"]
        class_name = result["class_name"]
        confidence = result["confidence"]
        cropped = result.get("cropped_image", None)

        col1, col2 = st.columns([2, 1])

        with col1:
            if cropped is not None:
                st.image(cropped, caption=f"Detected Sign {i+1}", use_column_width=True)
            else:
                st.warning("❌ No cropped image available.")

        with col2:
            st.markdown(f"<h2>Prediction Results</h2>", unsafe_allow_html=True)
            st.markdown(f"<div class='prediction-result'><b>Sign Type:</b> {class_name}</div>", 
                       unsafe_allow_html=True)
            st.markdown(f"<b>Confidence:</b> {confidence:.2%}", unsafe_allow_html=True)
            st.progress(float(confidence))
            
            if confidence > 0.79:
                certainty = "High Certainty"
                color = "var(--success-color)"
            elif confidence > 0.55:
                certainty = "Moderate Certainty"
                color = "var(--warning-color)"
            else:
                certainty = "Low Certainty"
                color = "red"
                
            st.markdown(f"<p style='color: {color}; font-weight: bold;'>{certainty}</p>", 
                       unsafe_allow_html=True)


def display_single_prediction(img_array, prediction_result):
    """
    Display prediction results from CNN-only pipeline (upload path)
    img_array: The original image array
    prediction_result: Tuple of (class_id, class_name, confidence)
    """
    class_id, class_name, confidence = prediction_result
    
    if class_id is None:
        st.warning("❌ No prediction available or error occurred.")
        return
        
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(img_array, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.markdown(f"<h2>Prediction Results</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-result'><b>Sign Type:</b> {class_name}</div>", 
                   unsafe_allow_html=True)
        st.markdown(f"<b>Confidence:</b> {confidence:.2%}", unsafe_allow_html=True)
        st.progress(float(confidence))
        
        if confidence > 0.79:
            certainty = "High Certainty"
            color = "var(--success-color)"
        elif confidence > 0.55:
            certainty = "Moderate Certainty"
            color = "var(--warning-color)"
        else:
            certainty = "Low Certainty"
            color = "red"
            
        st.markdown(f"<p style='color: {color}; font-weight: bold;'>{certainty}</p>", 
                   unsafe_allow_html=True)


def process_image_and_predict(image, model):
    try:
        # Preprocess_image function
        processed_img = preprocess_image(image)
        
        # Predict_image function
        class_id, confidence = predict_image(processed_img, model)
        
        # Get class name from the mapping
        class_name = SIGN_CLASSES.get(class_id, f"Unknown (Class {class_id})")
        
        return (class_id, class_name, confidence)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return (None, "Error in prediction", 0.0)

def detect_and_classify(img_array, cnn_model, yolo_model):
    try:
        results = yolo_model(img_array, verbose=False)[0]

        # Check if road signs were detected
        if len(results.boxes) == 0:
            st.warning("❌ No road sign detected in the image.")
            return []

        all_predictions = []

        # Loop all bounding boxes
        for box in results.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = box

            # Crop detected sign 
            cropped = img_array[y1:y2, x1:x2]

            # Preprocess and classify using CNN
            processed = preprocess_image(cropped)
            class_id, confidence = predict_image(processed, cnn_model)
            class_name = SIGN_CLASSES.get(class_id, f"Unknown (Class {class_id})")

            all_predictions.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": (x1, y1, x2, y2),
                "cropped_image": cropped  # Include the cropped image in results
            })
        return all_predictions
    
    except Exception as e:
        st.error(f"Error during detection/classification: {e}")
        return []
    
# EXIF Correction Function
def correct_exif_orientation(image):
    try:
        exif = image._getexif()
        if exif is not None:
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            if orientation_key is not None:
                orientation = exif.get(orientation_key, None)

                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    except Exception as e:
        print("Manual EXIF orientation correction skipped:", e)
# PIllow ImageOps EXIF Transpose
    # This is a fallback in case the manual correction fails
    try:
        image = ImageOps.exif_transpose(image)
    except Exception as e:
        print("ImageOps.exif_transpose fallback skipped:", e)

    return image
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


# Main code starts here
def main():
    models = load_models()
    if models is None:
        st.error("Failed to load required models. Please check that model files exist in the current directory.")
        return

    model = models['cnn']
    yolo_model = models['yolo']
    
    st.markdown("<h1 style='text-align: center;'>Traffic Sign Recognition System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='css-card'>
        <h3>About this App</h3>
        <p>This application uses computer vision to identify traffic signs from either:</p>
        <ul>
            <li>Live camera feed</li>
            <li>Uploaded image files (.png, .jpg, etc.)</li>
        </ul>
        <p>The system can recognize 43 different classes of traffic signs.</p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2 = st.tabs(["📷 Camera Input", "🖼️ Image Upload"])

    with tab1:
        st.markdown("<h2>Live Camera Recognition</h2>", unsafe_allow_html=True)
        st.markdown("Position a traffic sign in front of your camera and capture the image.")

        # Initialize session state for camera toggle
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False

        camera_col1, camera_col2 = st.columns([3, 1])

        with camera_col1:
            # Toggle camera on/off
            if st.button("Turn Camera On" if not st.session_state.camera_on else "Turn Camera Off"):
                st.session_state.camera_on = not st.session_state.camera_on
                st.rerun()

            if st.session_state.camera_on:
                camera_img = st.camera_input("Take a picture of a traffic sign", key="camera")

                if camera_img is not None:
                    # Load and correct image
                    image = Image.open(camera_img)
                    corrected_image = correct_exif_orientation(image)
                    img_array = np.array(corrected_image)

                    with st.spinner("🔍 Detecting and classifying traffic signs..."):
                        predictions = detect_and_classify(img_array, model, yolo_model)

                        # Check if any predictions were returned
                        if isinstance(predictions, list) and len(predictions) > 0:
                            display_prediction_results(predictions)
                        else:
                            st.warning("No traffic signs detected.")
        
        with camera_col2:
            st.markdown("""
            <div style='padding: 20px; background-color: var(--secondary-bg-color); border-radius: 10px;'>
                <h4>Instructions:</h4>
                <ul>
                    <li>Ensure good lighting</li>
                    <li>Center the sign in frame</li>
                    <li>Hold camera steady</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h2>Image Upload Recognition</h2>", unsafe_allow_html=True)
        st.markdown("Upload an image containing a traffic sign for recognition.")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Convert to PIL Image
            image = Image.open(uploaded_file)
            # Apply EXIF correction
            corrected_image = correct_exif_orientation(image)
            img_array = np.array(corrected_image)
            
            with st.spinner("Analyzing traffic sign..."):
                prediction_results = process_image_and_predict(img_array, model)
                
                # Display prediction using the single prediction display function
                display_single_prediction(img_array, prediction_results)
    
    # Add an FAQ section at the bottom
    with st.expander("ℹ️ FAQs about Traffic Sign Recognition"):
        st.markdown("""
        <h4>Frequently Asked Questions</h4>
        <p><b>Q: What types of traffic signs can this app recognize?</b><br>
        A: The system can identify 43 different classes of traffic signs including stop signs, speed limits, yield signs, and many more.</p>
        
        <p><b>Q: How accurate is the recognition?</b><br>
        A: The accuracy depends on various factors including image quality, lighting conditions, and sign visibility. For best results, ensure the sign is clearly visible and well-lit.Though the CNN's accuracy is 95.31%</p>
        
        <p><b>Q: Can I use images from any camera?</b><br>
        A: Yes, you can either use your device's camera directly through the app or upload images taken from any camera.</p>
        
        <p><b>Q: What image formats are supported?</b><br>
        A: The app supports common image formats including .jpg, .jpeg, and .png.</p>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style='text-align: center; margin-top: 30px; padding: 10px; font-size: 0.8em; color: #666;'>
        <p>© 2025 Traffic Sign Recognition System</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()