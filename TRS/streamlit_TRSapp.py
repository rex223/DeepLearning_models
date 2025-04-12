import nest_asyncio
nest_asyncio.apply()
import streamlit as st
st.set_option('server.fileWatcherType', 'none')

import cv2
import numpy as np
from PIL import Image
import io
import time
import tensorflow as tf
from ultralytics import YOLO
from preprocess_image import preprocess_image
from predict_image import *
import torch

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
    models = {}
    try:
        models['cnn'] = tf.keras.models.load_model('final_model.h5')
        models['yolo'] = YOLO('yolo_final.pt')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None


def display_prediction_results(image, prediction_results):
    class_id, class_name, confidence = prediction_results
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        try:
            st.image(image, caption="Traffic Sign", width=600)
        except Exception as e:
            st.error(f"Error displaying image: {e}")
    
    with col2:
        st.markdown("<div class='gradient-border'>", unsafe_allow_html=True)
        st.markdown(f"<h2>Prediction Results</h2>", unsafe_allow_html=True)
        st.markdown(f"<div class='prediction-result'><b>Sign Type:</b> {class_name}</div>", 
                   unsafe_allow_html=True)
        
        # Creating a progress bar for confidence
        st.markdown(f"<b>Confidence:</b> {confidence:.2%}", unsafe_allow_html=True)
        st.progress(float(confidence))
        
        if confidence > 0.9:
            certainty = "High Certainty"
            color = "var(--success-color)"
        elif confidence > 0.7:
            certainty = "Moderate Certainty"
            color = "var(--warning-color)"
        else:
            certainty = "Low Certainty"
            color = "red"
            
        st.markdown(f"<p style='color: {color}; font-weight: bold;'>{certainty}</p>", 
                   unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

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
        # Run YOLO detection
        results = yolo_model(img_array, verbose=False)[0]

        if len(results.boxes) == 0:
            st.warning("❌ No road sign detected in the image.")
            return (None, "No sign detected", 0.0)

        # Get the first detected bounding box
        box = results.boxes.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = box

        # Crop the detected sign from the image
        cropped = img_array[y1:y2, x1:x2]

        # Preprocess and classify using CNN
        processed = preprocess_image(cropped)
        class_id, confidence = predict_image(processed, cnn_model)
        class_name = SIGN_CLASSES.get(class_id, f"Unknown (Class {class_id})")

        return (class_id, class_name, confidence)

    except Exception as e:
        st.error(f"Error during detection/classification: {e}")
        return (None, "Detection Error", 0.0)
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
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📷 Camera Input", "🖼️ Image Upload"])

    with tab1:
        st.markdown("<h2>Live Camera Recognition</h2>", unsafe_allow_html=True)
        st.markdown("Position a traffic sign in front of your camera and capture the image.")
        
        # Initialize session state for camera toggle
        if 'camera_on' not in st.session_state:
            st.session_state.camera_on = False
        
        camera_col1, camera_col2 = st.columns([3, 1])
        
        with camera_col1:
            # Toggle button
            if st.button("Turn Camera On" if not st.session_state.camera_on else "Turn Camera Off"):
                st.session_state.camera_on = not st.session_state.camera_on
                st.rerun()  
            if st.session_state.camera_on:
                camera_img = st.camera_input("Take a picture of a traffic sign", key="camera")
                
                if camera_img is not None:
                    # Convert to PIL Image
                    image = Image.open(camera_img)
                    img_array = np.array(image)
                    
                    with st.spinner("Analyzing traffic sign..."):
                        prediction_results = detect_and_classify(img_array,model,yolo_model)
                        
                        # Display prediction
                        display_prediction_results(img_array, prediction_results)
        
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
            img_array = np.array(image)
            
            with st.spinner("Analyzing traffic sign..."):
                prediction_results = process_image_and_predict(img_array, model)
                
                # Display prediction
                display_prediction_results(img_array, prediction_results)
    
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