import streamlit as st
import cv2
import numpy as np
import time

# Title
st.title("Live AI Analysis Interface")

# Webcam Feed (Top Section)
st.header("Live Webcam Feed")

# Placeholder for the video feed
video_placeholder = st.empty()

# DroidCam URL (replace with your phone's DroidCam IP and port, e.g., http://192.168.1.100:4747/video)
DROIDCAM_URL = "http://192.168.0.133:4747/video"  # Update this with your DroidCam IP

# Initialize video capture with DroidCam
cap = cv2.VideoCapture(DROIDCAM_URL)

if not cap.isOpened():
    st.error("Failed to connect to DroidCam. Please check the IP address and ensure DroidCam is running on your phone.")
else:
    # Display the video feed
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to retrieve frame from DroidCam. Retrying...")
            time.sleep(1)
            continue

        # Convert frame to RGB (OpenCV uses BGR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame in Streamlit
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        # Small delay to avoid overloading
        time.sleep(0.03)  # Approximately 30 FPS

# AI Analysis Toggle
st.header("AI Analysis Controls")
ai_toggle = st.toggle("Toggle AI Analysis (10-15s)", value=False)
if ai_toggle:
    st.write("AI is analyzing the live feed for 10-15 seconds...")

# TTS Toggle
st.header("Text-to-Speech Controls")
tts_toggle = st.toggle("Toggle TTS (AI Avatar Output)", value=False)
if tts_toggle:
    st.write("DeepMotion avatar is reading out LLM output...")

# DeepMotion Avatar Placeholder (Bottom Section)
st.header("DeepMotion Avatar Feedback")
if tts_toggle:
    st.write("Placeholder: DeepMotion avatar would display here with TTS output.")
else:
    st.write("Placeholder: Avatar inactive.")

# Release the capture when done (though Streamlit may not reach this in a running app)
cap.release()