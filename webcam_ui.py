import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time

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

# Define function to calculate angle between limbs
def calculate_angle(a, b, c):
    a = np.array(a)     # First point
    b = np.array(b)     # Middle point
    c = np.array(c)     # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    degrees = np.abs(radians * 180 / np.pi)

    if degrees > 180.0:
        degrees = 360 - degrees

    return degrees


# Initialize video capture with DroidCam
cap = cv2.VideoCapture(0)    # DROIDCAM_URL

# fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'mp4v' | 'XVID' | 'avc1
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
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

            # Display the frame
            # out.write(frame)
            # cv2.imshow('Workout Window', frame)

            # Check for 'q' key press
            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                print('Stopping live feed.')
                break

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