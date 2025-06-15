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
import io
import uuid

from utilities import upload_and_process_video, generate_content_from_video, calculate_angle, calculate_distance, is_inside_zone, left_arm_bicep_curl, right_arm_bicep_curl, both_arms_bicep_curl, both_arms_lateral_raise

# Set wide layout to maximize space
st.set_page_config(layout="wide")

load_dotenv('.env')

# Load API keys
gemini_apikey = os.getenv("GEMINI_APIKEY")
elevenlabs_apikey = os.getenv("ELEVENLABS_APIKEY")

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
processing_started_at = None  # Time when processing started
counter = 0  # Rep counter
left_stage = None  # Left arm stage (None, "up", "down")
right_stage = None  # Right arm stage (None, "up", "down")
current_exercise = "Left Arm Bicep Curl"  # Default exercise
last_exercise = current_exercise  # Track last valid exercise

# Lock for video writer access
video_writer_lock = threading.Lock()

# Exercise function mapping
exercise_functions = {
    "Left Arm Bicep Curl": left_arm_bicep_curl,
    "Right Arm Bicep Curl": right_arm_bicep_curl,
    "Both Arms Bicep Curl": both_arms_bicep_curl,
    "Both Arms Lateral Raise": both_arms_lateral_raise
}

# Function to handle stop recording
def stop_recording():
    global recording
    recording = False
    video_placeholder.empty()

# Function to reset sampling
def reset_sampling():
    global reset_flag, segment_start_time, frame_count, video_writer, processing_started_at, counter, left_stage, right_stage
    reset_flag = True
    segment_start_time = time.time()
    frame_count = 0
    counter = 0
    left_stage = None
    right_stage = None
    with video_writer_lock:
        if video_writer:
            video_writer.release()
            video_writer = None
    processing_started_at = None

# Function to set exercise
def set_exercise(exercise):
    global current_exercise, last_exercise, counter, left_stage, right_stage
    if exercise != current_exercise:
        current_exercise = exercise
        last_exercise = exercise
        counter = 0
        left_stage = None
        right_stage = None

# Sidebar for controls
with st.sidebar:
    st.markdown("🛠️ Controls")
    tts_toggle = st.toggle("🔊 TTS", value=False)
    if tts_toggle:
        st.write("Text-to-speech enabled.")
    if st.button("Switch to Chatbot"):
        st.switch_page("chatbot.py")
    st.button("⏹️ Stop", key="stop", on_click=stop_recording)

# Two columns for camera and feedback
col1, col2 = st.columns([7, 5])

# Camera (Left Column)
with col1:
    st.markdown("📸 Camera")
    # Toggles and Reset button
    btn_col1, btn_col2, btn_col3 = st.columns(3)
    with btn_col1:
        st.button("🔄 Reset Sampling", key="reset", on_click=reset_sampling)
    with btn_col2:
        ai_toggle = st.toggle("🧠 AI Analysis", value=False)
    with btn_col3:
        selection_mode = st.toggle("🤖 AI Detection", value=True)
    exercise_placeholder = st.empty()
    # Exercise selection buttons in a single row
    st.markdown("Select Exercise:", help="Click a button to manually select an exercise.")
    btn_ex1, btn_ex2, btn_ex3, btn_ex4 = st.columns(4)
    with btn_ex1:
        if st.button("Left Bicep", key="ex_left_bicep"):
            set_exercise("Left Arm Bicep Curl")
    with btn_ex2:
        if st.button("Right Bicep", key="ex_right_bicep"):
            set_exercise("Right Arm Bicep Curl")
    with btn_ex3:
        if st.button("Both Bicep", key="ex_both_bicep"):
            set_exercise("Both Arms Bicep Curl")
    with btn_ex4:
        if st.button("Lateral Raise", key="ex_lateral_raise"):
            set_exercise("Both Arms Lateral Raise")
    video_placeholder = st.empty()
    timer_placeholder = st.empty()

# AI Feedback (Right Column)
with col2:
    st.markdown("💬 Feedback")
    feedback_placeholder = st.empty()
    processing_placeholder = st.empty()

# Utility functions
def safe_delete_file(file_path, max_attempts=5, delay=1):
    for attempt in range(max_attempts):
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
                return True
            return True
        except (PermissionError, OSError):
            time.sleep(delay)
    st.error(f"Failed to delete temporary file {file_path} after {max_attempts} attempts.")
    print(f"Failed to delete file: {file_path}")
    return False

def text_to_speech(text):
    try:
        print(f"TTS processing text: {text}")
        client = ElevenLabs(api_key=elevenlabs_apikey)
        audio_stream = client.generate(
            text=text,
            voice="JBFqnCBsd6RMkjVDRZzb",
            model="eleven_multilingual_v2",
            output_format="mp3_44100_128"
        )
        # Convert generator to bytes
        audio_bytes = b''.join(audio_stream)
        print(f"TTS audio generated, size: {len(audio_bytes)} bytes")
        # Play audio using BytesIO
        audio_io = io.BytesIO(audio_bytes)
        play(audio_io)
        print("TTS audio played successfully")
    except Exception as e:
        st.error(f"Error in TTS: {e}")
        print(f"TTS error: {e}")

# Function to capture and process video segments
def capture_and_process_video(frame_queue):
    global recording, processing_status, segment_start_time, reset_flag, frame_count, video_writer, ai_toggle, processing_started_at, counter, left_stage, right_stage, current_exercise
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to connect to webcam. Please check your camera setup.")
        recording = False
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_dir = tempfile.gettempdir()
    segment_duration = 20  # Reduced for faster AI updates
    frame_rate = 20

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened() and recording:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to retrieve frame. Retrying...")
                time.sleep(1)
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = pose.process(rgb_frame)
            rgb_frame.flags.writeable = True
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with video_writer_lock:
                if frame_count == 0 and ai_toggle and not processing_status:
                    unique_id = str(uuid.uuid4())
                    temp_file_path = os.path.join(temp_dir, f"video_{unique_id}.mp4")
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4', dir=temp_dir) as temp_file:
                        temp_file_path = temp_file.name
                    video_writer = cv2.VideoWriter(temp_file_path, fourcc, frame_rate, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                    print(f"Starting new video segment: {temp_file_path}")

                if video_writer and ai_toggle and not processing_status:
                    video_writer.write(frame)
                    frame_count += 1
                elif video_writer and (not ai_toggle or processing_status):
                    video_writer.release()
                    video_writer = None
                    frame_count = 0
                    print("Stopped recording segment (AI toggle off or processing started).")

            # Process landmarks
            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    # Use direct indices instead of mp_pose.PoseLandmark
                    left_shoulder = [landmarks[11].x, landmarks[11].y]  # LEFT_SHOULDER
                    left_elbow = [landmarks[13].x, landmarks[13].y]     # LEFT_ELBOW
                    left_wrist = [landmarks[15].x, landmarks[15].y]     # LEFT_WRIST
                    h, w, _ = frame.shape
                    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
                    cv2.putText(frame_rgb, f'Angle: {angle:.1f}', tuple(np.multiply(left_elbow, [w, h]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                    # Rep counting
                    selected_function = exercise_functions.get(current_exercise, left_arm_bicep_curl)
                    mapping_dict = selected_function(landmarks, frame_rgb, h, w)
                    left_wrist = mapping_dict.get('left_wrist')
                    right_wrist = mapping_dict.get('right_wrist')
                    left_top = mapping_dict.get('left_top_zone')
                    right_top = mapping_dict.get('right_top_zone')
                    left_bottom = mapping_dict.get('left_bottom_zone')
                    right_bottom = mapping_dict.get('right_bottom_zone')

                    # Debug zone detection
                    print(f"Exercise: {current_exercise}, Left wrist: {left_wrist}, Left top: {left_top}, Left bottom: {left_bottom}")
                    print(f"Right wrist: {right_wrist}, Right top: {right_top}, Right bottom: {right_bottom}")

                    # Left arm logic
                    if left_wrist and left_bottom and left_top:
                        left_down = is_inside_zone(left_wrist, left_bottom)
                        left_up = is_inside_zone(left_wrist, left_top)
                        print(f"Left down: {left_down}, Left up: {left_up}, Left stage: {left_stage}")
                        if left_down and left_stage != 'down':
                            left_stage = 'down'
                            print("Left arm: Down stage detected")
                        elif left_stage == 'down' and left_up:
                            counter += 1
                            left_stage = 'up'
                            print(f"Left arm: Rep {counter} completed")
                            if tts_toggle:
                                threading.Thread(target=text_to_speech, args=(f"Rep {counter} completed!",), daemon=True).start()

                    # Right arm logic
                    if right_wrist and right_bottom and right_top:
                        right_down = is_inside_zone(right_wrist, right_bottom)
                        right_up = is_inside_zone(right_wrist, right_top)
                        print(f"Right down: {right_down}, Right up: {right_up}, Right stage: {right_stage}")
                        if right_down and right_stage != 'down':
                            right_stage = 'down'
                            print("Right arm: Down stage detected")
                        elif right_stage == 'down' and right_up:
                            counter += 1
                            right_stage = 'up'
                            print(f"Right arm: Rep {counter} completed")
                            if tts_toggle:
                                threading.Thread(target=text_to_speech, args=(f"Rep {counter} completed!",), daemon=True).start()

                # Display rep counter (always show, even if no landmarks)
                cv2.putText(frame_rgb, f'Reps: {counter}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                print(f"Displayed Reps: {counter}")

            except Exception as e:
                print(f"Error processing landmarks: {e}")

            mp_drawing.draw_landmarks(
                frame_rgb,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )

            try:
                frame_queue.put_nowait(frame_rgb)
            except queue.Full:
                pass

            if (ai_toggle and not processing_status and (time.time() - segment_start_time) >= segment_duration) or reset_flag:
                with video_writer_lock:
                    if video_writer:
                        video_writer.release()
                        video_writer = None
                        temp_file_path_local = temp_file_path

                if ai_toggle and not reset_flag:
                    processing_started_at = time.time()
                    threading.Thread(target=process_video_segment, args=(temp_file_path_local, f"segment_{int(time.time())}",), daemon=True).start()
                else:
                    if 'temp_file_path_local' in locals() and os.path.exists(temp_file_path_local):
                        safe_delete_file(temp_file_path_local)

                frame_count = 0
                segment_start_time = time.time()
                reset_flag = False

            time.sleep(0.05)

    cap.release()
    with video_writer_lock:
        if video_writer:
            video_writer.release()
    print("Webcam and video writer released. Thread terminated.")

# Function to process video segment and get feedback
def process_video_segment(file_path, display_name):
    global processing_status, processing_started_at, current_exercise, last_exercise
    processing_status = True
    print(f"AI processing started for {display_name}: {file_path}")

    # Verify file is an MP4
    if not file_path.endswith('.mp4') or not os.path.exists(file_path):
        st.error(f"Invalid video file: {file_path}")
        print(f"Invalid video file: {file_path}")
        processing_status = False
        processing_started_at = None
        return

    video_file = upload_and_process_video(file_path, display_name)
    if video_file:
        # Generate feedback
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
            if tts_toggle:
                threading.Thread(target=text_to_speech, args=(feedback,), daemon=True).start()

        # Detect exercise
        if selection_mode:
            try:
                with open('exercise_detection_prompt.txt', 'r') as f:
                    exercise_prompt = f.read()
            except FileNotFoundError:
                exercise_prompt = """You are provided with a 20-second video of a user performing a workout. Identify the specific exercise by analyzing arm movements and body posture. Choose exactly one exercise from the following options, returning only the exercise name:

                - Left Arm Bicep Curl: The left arm bends at the elbow, moving the hand toward the shoulder, while the right arm remains relatively stationary.
                - Right Arm Bicep Curl: The right arm bends at the elbow, moving the hand toward the shoulder, while the left arm remains relatively stationary.
                - Both Arms Bicep Curl: Both arms bend at the elbows simultaneously, moving both hands toward the shoulders.
                - Both Arms Lateral Raise: Both arms are raised outward to shoulder height, forming a T-shape with the body, then lowered back to the sides.

                Ensure the response contains only the exercise name, with no additional text or formatting."""
            exercise = generate_content_from_video(video_file, exercise_prompt)
            print(f"Detected exercise: '{exercise}'")
            if exercise in exercise_functions:
                current_exercise = exercise
                last_exercise = exercise
            else:
                st.warning(f"AI exercise detection failed: '{exercise}'. Using last known exercise: {last_exercise}")
                current_exercise = last_exercise

        genai.delete_file(video_file.name)
    else:
        print(f"Video upload failed for {file_path}")

    safe_delete_file(file_path)
    processing_status = False
    processing_started_at = None
    print(f"AI processing finished for {display_name}")

# Main app logic
frame_queue = queue.Queue(maxsize=10)

# Start capture thread
capture_thread = threading.Thread(target=capture_and_process_video, args=(frame_queue,), daemon=True)
capture_thread.start()

# Update Streamlit UI
while recording:
    try:
        frame_rgb = frame_queue.get(timeout=0.1)
        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

        # Update exercise selection/display
        if not selection_mode:
            exercise = exercise_placeholder.selectbox(
                "Select Exercise",
                ["Left Arm Bicep Curl", "Right Arm Bicep Curl", "Both Arms Bicep Curl", "Both Arms Lateral Raise"],
                key=f"exercise_select_{time.time()}"
            )
            if exercise != current_exercise:
                set_exercise(exercise)
        else:
            exercise_placeholder.markdown(f"Detected Exercise: {current_exercise}")

        # Update timer
        if ai_toggle:
            current_time = time.time()
            if processing_status:
                timer_placeholder.markdown("AI Analyzing Video...")
                timer_placeholder.progress(0.0)
            else:
                remaining_time = max(0, 20 - (current_time - segment_start_time))
                timer_placeholder.markdown(f"⏲️ Sampling: {int(remaining_time)}s remaining")
                timer_placeholder.progress(remaining_time / 20.0)
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
        except queue.Empty:
            pass
    except queue.Empty:
        pass
    time.sleep(0.01)

# Clean up
recording = False