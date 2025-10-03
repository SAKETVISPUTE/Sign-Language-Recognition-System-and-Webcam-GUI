# app.py

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Set page configuration
st.set_page_config(page_title="ASL Interpreter", page_icon="✋", layout="wide")

# Load the trained model using Streamlit's caching for efficiency
@st.cache_resource
def load_asl_model():
    """Loads the pre-trained ASL CNN model."""
    try:
        model = tf.keras.models.load_model('asl_cnn_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the 'asl_cnn_model.h5' file is in the same directory and you have run the training script.")
        return None

model = load_asl_model()

# Create a mapping from index to letter
# Note: The dataset labels are 0-24, but 'J' (index 9) is skipped
label_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# --- Streamlit App Layout ---
st.title("Live ASL Sign Language Interpreter 🤟")
st.write("This app uses your webcam to interpret American Sign Language alphabet gestures in real-time.")

col1, col2 = st.columns(2)

with col1:
    st.header("Instructions")
    st.info(
        """
        1.  Click the **Start Webcam** button below.
        2.  Allow the browser to access your camera.
        3.  Place your hand inside the **green rectangle (ROI)**.
        4.  The model will predict the letter and display it.
        5.  Click **Stop Webcam** to end the stream.
        """
    )
    # Buttons to control the webcam
    start_button = st.button("Start Webcam", key="start")
    stop_button = st.button("Stop Webcam", key="stop")

with col2:
    st.header("Webcam Feed")
    # Placeholder for the video feed
    FRAME_WINDOW = st.image([])

# --- Webcam and Inference Logic ---
if start_button:
    st.session_state.run = True
if stop_button:
    st.session_state.run = False

if 'run' not in st.session_state:
    st.session_state.run = False

cap = cv2.VideoCapture(0)

while st.session_state.run:
    ret, frame = cap.read()
    if not ret:
        st.write("The video stream has ended.")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)
    
    # Define Region of Interest (ROI)
    roi_x1, roi_y1, roi_x2, roi_y2 = 100, 100, 328, 328 # 228x228 ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Extract ROI and preprocess it
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized_roi = cv2.resize(gray_roi, (28, 28))
    normalized_roi = resized_roi / 255.0
    input_data = np.reshape(normalized_roi, (1, 28, 28, 1))

    prediction_text = "No prediction"
    if model is not None:
        # Make a prediction
        prediction = model.predict(input_data, verbose=0)
        predicted_index = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Get the letter from the mapping
        predicted_letter = label_mapping.get(predicted_index, "?")
        
        # Prepare text to display
        prediction_text = f"Prediction: {predicted_letter} ({confidence:.1f}%)"

    # Display the prediction text on the frame
    cv2.putText(frame, prediction_text, (roi_x1, roi_y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Convert the frame to RGB for Streamlit display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Update the placeholder with the new frame
    FRAME_WINDOW.image(frame_rgb)

cap.release()

if not st.session_state.run:
    st.warning("Webcam is off. Click 'Start Webcam' to begin.")