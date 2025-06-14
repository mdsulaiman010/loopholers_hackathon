import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
import os
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import play
import threading
import queue
from utilities import upload_and_process_video, generate_content_from_video, calculate_angle

# Set wide layout to maximize space
st.set_page_config(layout="wide")

load_dotenv('.env')

# Load API keys
gemini_apikey = os.getenv("GEMINI_APIKEY")
elevenlabs_apikey = os.getenv("ELEVENLABS_APIKEY")
# Note: heygen_apikey is loaded but not used

genai.configure(api_key=gemini_apikey)

# Initialize model and drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Global variables
recording = True  # Always-on webcam
feedback_queue = queue.Queue()  # Global feedback queue

# Function to handle stop recording
def stop_recording():
    global recording
    recording = False
    video_placeholder.empty()

# Title (optional, can be removed)
st.title("Live AI Analysis Interface")

# Sidebar for toggles and stop button
with st.sidebar:
    st.markdown("🛠️ Controls")
    ai_toggle = st.toggle("🧠 AI Analysis", value=False)
    if ai_toggle:
        st.write("Analyzing 30-second clips...")
    tts_toggle = st.toggle("🔊 TTS (Avatar)", value=False)
    if tts_toggle:
        st.write("Avatar reading LLM output...")
    st.button("⏹️ Stop", key="stop", on_click=stop_recording)

# Three columns for camera, feedback, and avatar
col1, col2, col3 = st.columns([7, 3, 2])  # Wider columns to maximize space

# Camera (Left Column)
with col1:
    st.markdown("📸 Camera")
    video_placeholder = st.empty()

# AI Feedback (Middle Column)
with col2:
    st.markdown("💬 Feedback")
    feedback_placeholder = st.empty()

# Avatar (Right Column)
with col3:
    st.markdown("🧑‍🎨 Avatar")
    if tts_toggle:
        st.write("Avatar active with TTS.")
    else:
        st.write("Avatar inactive.")

# Fallback utility functions (remove if defined in utilities)
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    degrees = np.abs(radians * 180 / np.pi)
    if degrees > 180.0:
        degrees = 360 - degrees
    return degrees

def upload_and_process_video(file_path, display_name):
    try:
        video_file = genai.upload_file(path=file_path, display_name=display_name, mime_type="video/mp4")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)
        if video_file.state.name != "ACTIVE":
            st.error(f"File {video_file.name} is in {video_file.state.name} state. Re-uploading...")
            genai.delete_file(video_file.name)
            video_file = genai.upload_file(path=file_path, display_name=display_name, mime_type="video/mp4")
            while video_file.state.name == "PROCESSING":
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
        if video_file.state.name == "ACTIVE":
            return video_file
        else:
            st.error(f"File {video_file.name} failed to reach ACTIVE state.")
            return None
    except Exception as e:
        st.error(f"Error uploading video: {e}")
        return None

def generate_content_from_video(video_file, prompt):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        return response.text
    except Exception as e:
        st.error(f"Error generating content: {e}")
        return None

def safe_delete_file(file_path, max_attempts=5, delay=1):
    for attempt in range(max_attempts):
        try:
            os.remove(file_path)
            return True
        except PermissionError:
            time.sleep(delay)
    st.error(f"Failed to delete temporary file {file_path} after {max_attempts} attempts.")
    return False

def text_to_speech(text):
    try:
        client = ElevenLabs(api_key=elevenlabs_apikey)
        audio = client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        play(audio)
    except Exception as e:
        st.error(f"Error in TTS: {e}")

# Function to capture and process video segments
def capture_and_process_video(frame_queue):
    global recording  # Declare global variable
    cap = cv2.VideoCapture(0)  # Webcam
    if not cap.isOpened():
        st.error("Failed to connect to webcam. Please check your camera setup.")
        recording = False
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_dir = tempfile.gettempdir()
    segment_duration = 30  # 30-second clips
    frame_rate = 20
    frame_count = 0
    video_writer = None
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and recording:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to retrieve frame. Retrying...")
                time.sleep(1)
                continue

            # Convert frame to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Initialize video writer for new segment
            if frame_count == 0:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
                video_writer = cv2.VideoWriter(temp_file.name, fourcc, frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

            # Write frame to video
            video_writer.write(frame)

            # Process landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                h, w, _ = frame.shape
                angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                cv2.putText(frame, f'Angle: {angle:.1f}', tuple(np.multiply(left_elbow, [w, h]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                print(f"Left shoulder: {left_shoulder}")
                print(f"Angle: {angle:.1f} degrees")
            except Exception as e:
                print(f"Error processing landmarks: {e}")

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame_rgb, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            # Add frame to queue for display
            try:
                frame_queue.put_nowait(frame_rgb)
            except queue.Full:
                pass

            frame_count += 1
            if (time.time() - start_time) >= segment_duration and ai_toggle:
                video_writer.release()
                video_writer = None
                threading.Thread(target=process_video_segment, args=(temp_file.name, f"segment_{int(time.time())}"), daemon=True).start()
                frame_count = 0
                start_time = time.time()

            time.sleep(0.05)  # Approximately 20 FPS

    cap.release()
    if video_writer:
        video_writer.release()

# Function to process video segment and get feedback
def process_video_segment(file_path, display_name):
    video_file = upload_and_process_video(file_path, display_name)
    if video_file:
        try:
            with open('livefeed-sys-prompt.txt', 'r') as f:
                prompt = f.read()
        except FileNotFoundError:
            prompt = """You will be provided a video of the user doing some kind of workout.
            Your job is to provide feedback on the provided video and give insights on how to improve their form and technique.
            Ensure your points are helpful and easy to grasp."""
        feedback = generate_content_from_video(video_file, prompt)
        if feedback:
            feedback_queue.put(feedback)
        genai.delete_file(video_file.name)  # Clean up file from Gemini
    safe_delete_file(file_path)

# Main app logic
frame_queue = queue.Queue(maxsize=10)

# Start capture thread
capture_thread = threading.Thread(target=capture_and_process_video, args=(frame_queue,), daemon=True)
capture_thread.start()

# Update Streamlit UI
while recording:
    try:
        # Update video feed
        frame_rgb = frame_queue.get(timeout=0.1)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
        
        # Update feedback
        try:
            feedback = feedback_queue.get_nowait()
            feedback_placeholder.text(f"Gemini Feedback: {feedback}")
            if tts_toggle:
                text_to_speech(feedback)
        except queue.Empty:
            pass
    except queue.Empty:
        pass
    time.sleep(0.01)

# Clean up
recording = False