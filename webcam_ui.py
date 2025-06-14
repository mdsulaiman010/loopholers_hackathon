import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os 
from dotenv import load_dotenv
import google.generativeai as genai
from utils import upload_and_process_video, generate_content_from_video, calculate_angle

# Load relevant API keys
load_dotenv('.env')
gemini_apikey = os.environ["GEMINI_APIKEY"]
heygen_apikey = os.environ['HEYGEN_APIKEY']
elevenlabs_apikey = os.environ['ELEVENLABS_APIKEY']

genai.configure(api_key=gemini_apikey)

# Initialize model and drawing utilities
mp_drawing = mp.solutions.drawing_utils # Add landmarks and drawing elements to video
mp_pose = mp.solutions.pose

# Title
st.title("Live AI Analysis Interface")

# Webcam Feed (Top Section)
st.header("Live Webcam Feed")

# Placeholder for the video feed
video_placeholder = st.empty()

# DroidCam URL (replace with your phone's DroidCam IP and port, e.g., http://192.168.1.100:4747/video)
DROIDCAM_URL = "http://192.168.0.133:4747/video"  # Update this with your DroidCam IP


if 'recording' not in st.session_state:
    st.session_state.recording = False

col1, col2 = st.columns(2)
with col1: start_record = st.button('Start recording', key='start')
with col2: stop_record = st.button('Stop recording', key='stop')


# Initialize video capture with DroidCam
if start_record:
    cap = cv2.VideoCapture(0)    # DROIDCAM_URL

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Or 'mp4v' | 'XVID' | 'avc1
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        if not cap.isOpened():
            st.error("Failed to connect to DroidCam. Please check the IP address and ensure DroidCam is running on your phone.")
        else:
            # Display the video feed
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to retrieve frame from DroidCam. Retrying...")
                    time.sleep(1)
                    continue
                
                # Convert frame from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Set frame flags to False (optimization)
                rgb_frame.flags.writeable = False

                # Make detection predictions
                results = pose.process(rgb_frame)

                # Set frame memory back to True
                rgb_frame.flags.writeable = True

                # Convert frame to RGB (OpenCV uses BGR)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Small delay to avoid overloading
                time.sleep(0.03)  # Approximately 30 FPS

                try:
                    landmarks = results.pose_landmarks.landmark

                    # Get x,y coordinates of points (normalized 0-1)
                    left_shoulder = [
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y
                    ]
                    left_elbow = [
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y
                    ]
                    left_wrist = [
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y
                    ]

                    # Get actual frame size
                    h, w, _ = frame.shape
                    
                    # Calculate angle between points
                    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

                    # Display angle on frame (use original BGR frame for display)
                    cv2.putText(frame, 
                                f'Angle: {angle:.1f}', 
                                tuple(np.multiply(left_elbow, [w, h]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Print shoulder coordinates (this should now work)
                    print(f"Left shoulder: {left_shoulder}")
                    print(f"Angle: {angle:.1f} degrees")

                except Exception as e:
                    print(f"Error processing landmarks: {e}")

                # Draw landmarks and connections on the original BGR frame
                mp_drawing.draw_landmarks(
                    frame_rgb, 
                    results.pose_landmarks, 
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Display the frame in Streamlit
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                out.write(frame)

                if stop_record:
                    print('Stopping feed')
                    st.session_state.recording = False
                    break
        cap.release()

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


video_file_path = "output.mp4"
display_name = "sample_video"

# Check if the file exists
if not os.path.exists(video_file_path):
    print(f"Video file {video_file_path} not found.")

# Load in system prompt
with open('livefeed-sys-prompt.txt', 'r') as f:
    system_prompt = f.read()

# Check if the file is already uploaded
try:
    file_list = genai.list_files(page_size=100)
    video_file = next((f for f in file_list if f.display_name == display_name), None)

    if video_file:
        print(f"Using existing file: {video_file.uri}")
    else:
        video_file = upload_and_process_video(video_file_path, display_name)

    if video_file:
        response_text = generate_content_from_video(video_file, system_prompt)

        if response_text:
            print("\nGenerated Response:")
            print(response_text)
        else:
            print("Failed to generate response.")
    else:
        print("Failed to upload or process video.")
except Exception as e:
    print(f"Error: {e}")



# Release the capture when done (though Streamlit may not reach this in a running app)
# cap.release()