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
processing_status = False  # AI processing indicator
segment_start_time = time.time()  # Timer start
reset_flag = False  # Reset sampling flag
frame_count = 0  # Frame counter for reset
video_writer = None  # Video writer for reset
# New global variable to hold the time when processing started, if any
processing_started_at = None

# Function to handle stop recording
def stop_recording():
    global recording
    recording = False
    video_placeholder.empty()

# Function to reset sampling
def reset_sampling():
    global reset_flag, segment_start_time, frame_count, video_writer, processing_started_at
    reset_flag = True
    segment_start_time = time.time()
    frame_count = 0
    if video_writer:
        video_writer.release()
        video_writer = None
    processing_started_at = None # Reset processing state as well

# Title (optional, can be removed)
st.title("Live AI Analysis Interface")

# Sidebar for controls (only TTS toggle and Stop button remain here)
with st.sidebar:
    st.markdown("🛠️ Controls")
    # AI Analysis toggle has been moved
    tts_toggle = st.toggle("🔊 TTS (Avatar)", value=False)
    if tts_toggle:
        st.write("Avatar reading LLM output...")
    st.button("⏹️ Stop", key="stop", on_click=stop_recording)

# Three columns for camera, feedback, and avatar
col1, col2, col3 = st.columns([7, 3, 2])  # Wider columns to maximize space

# Camera (Left Column)
with col1:
    st.markdown("📸 Camera")
    # Create two columns for the buttons/toggles
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        st.button("🔄 Reset Sampling", key="reset", on_click=reset_sampling)
    with btn_col2:
        ai_toggle = st.toggle("🧠 AI Analysis", value=False)
        # The message "Analyzing 30-second clips..." can be placed directly below the toggle if desired,
        # but will appear on a new line due to Streamlit's layout.
    if ai_toggle:
        st.write("Analyzing 30-second clips...")
    # The video placeholder and timer remain below the buttons/toggles
    video_placeholder = st.empty()
    timer_placeholder = st.empty()  # Timer display


# AI Feedback (Middle Column)
with col2:
    st.markdown("💬 Feedback")
    feedback_placeholder = st.empty()
    processing_placeholder = st.empty()  # Processing indicator

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
    global recording, processing_status, segment_start_time, reset_flag, frame_count, video_writer, ai_toggle, processing_started_at
    cap = cv2.VideoCapture(0)  # Webcam
    if not cap.isOpened():
        st.error("Failed to connect to webcam. Please check your camera setup.")
        recording = False
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_dir = tempfile.gettempdir()
    segment_duration = 30  # 30-second clips
    frame_rate = 20

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
            # Only start a new video segment if AI is toggled AND not currently processing
            if frame_count == 0 and ai_toggle and not processing_status:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir)
                video_writer = cv2.VideoWriter(temp_file.name, fourcc, frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                print(f"Starting new video segment: {temp_file.name}") # Debug print

            # Write frame to video ONLY if video_writer is active (i.e., not processing and ai_toggle is on)
            if video_writer and ai_toggle and not processing_status:
                video_writer.write(frame)
                frame_count += 1
            elif video_writer and (not ai_toggle or processing_status):
                # If AI toggle is off or processing has started, release the current writer
                # and prepare for next segment.
                video_writer.release()
                video_writer = None
                frame_count = 0 # Reset frame count so it restarts fresh when conditions are met
                print("Stopped recording segment (AI toggle off or processing started).")


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
                # print(f"Left shoulder: {left_shoulder}") # Too verbose
                # print(f"Angle: {angle:.1f} degrees") # Too verbose
            except Exception as e:
                # print(f"Error processing landmarks: {e}") # Debug print if needed
                pass

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

            # Check if segment duration is reached or reset is requested
            # Only trigger AI if ai_toggle is ON and not already processing
            if (ai_toggle and not processing_status and (time.time() - segment_start_time) >= segment_duration) or reset_flag:
                if video_writer:
                    video_writer.release()
                    video_writer = None # Ensure it's set to None after releasing
                
                # If triggered by time or reset and AI is toggled on, start processing
                if ai_toggle and not reset_flag: # Only trigger AI if AI is on and it's not a reset
                    # Important: Signal that processing has started
                    processing_started_at = time.time() # Capture the time processing started
                    threading.Thread(target=process_video_segment, args=(temp_file.name, f"segment_{int(time.time())}",), daemon=True).start()
                else: # This is a reset or AI is off, so just clean up and prepare for next segment
                    if 'temp_file' in locals() and os.path.exists(temp_file.name):
                        safe_delete_file(temp_file.name)

                frame_count = 0
                segment_start_time = time.time() # Reset timer for next segment
                reset_flag = False

            time.sleep(0.05)  # Approximately 20 FPS

    cap.release()
    if video_writer:
        video_writer.release()
    print("Webcam and video writer released. Thread terminated.")


# Function to process video segment and get feedback
def process_video_segment(file_path, display_name):
    global processing_status, processing_started_at
    processing_status = True # Set global processing status to True
    print(f"AI processing started for {display_name}") # Debug print

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
            # Check tts_toggle here (it's a global variable from main thread)
            if tts_toggle:
                threading.Thread(target=text_to_speech, args=(feedback,), daemon=True).start()
        genai.delete_file(video_file.name)  # Clean up file from Gemini
    safe_delete_file(file_path)

    processing_status = False # Set global processing status back to False
    processing_started_at = None # Reset the processing start time
    print(f"AI processing finished for {display_name}") # Debug print


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

        # Update timer
        if ai_toggle:
            current_time = time.time()
            if processing_status:
                # If processing, freeze the displayed time
                # The segment_start_time represents when the *current* segment began for recording.
                # When processing starts, we want to maintain the remaining time from that point.
                # However, your current segment_start_time is reset *after* processing starts.
                # Let's adjust this to correctly reflect the remaining time for the *next* segment.
                # Simplest is to just show "AI Analyzing..." instead of a timer during processing.
                remaining_time_for_display = 0 # Or just not show it.
                timer_placeholder.markdown("AI Analyzing Video...")
                timer_placeholder.progress(0.0) # Set progress bar to empty or full
            else:
                remaining_time = max(0, 30 - (current_time - segment_start_time))
                timer_placeholder.markdown(f"⏲️ Sampling: {int(remaining_time)}s remaining")
                timer_placeholder.progress(remaining_time / 30.0)
        else:
            timer_placeholder.empty()

        # Update processing indicator
        if processing_status:
            processing_placeholder.markdown("🕒 AI Processing…")
        else:
            processing_placeholder.empty()

        # Update feedback
        try:
            feedback = feedback_queue.get_nowait()
            feedback_placeholder.text(f"Gemini Feedback: {feedback}")
            # TTS is now triggered from process_video_segment function (if tts_toggle is true there)
        except queue.Empty:
            pass
    except queue.Empty:
        pass
    time.sleep(0.01)

# Clean up
recording = False