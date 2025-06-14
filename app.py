import cv2
import mediapipe as mp
import numpy as np
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv
load_dotenv('.env')

mp_drawing = mp.solutions.drawing_utils # Add landmarks and drawing elements to video
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)     # First point
    b = np.array(b)     # Middle point
    c = np.array(c)     # End point

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    degrees = np.abs(radians * 180 / np.pi)

    if degrees > 180.0:
        degrees = 360 - degrees

    return degrees 

capture = cv2.VideoCapture(1)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Or 'mp4v' | 'XVID' | 'avc1
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    print("Starting pose detection... Press 'q' to quit")
    
    while capture.isOpened():
        # Read current frame
        ret, frame = capture.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set frame flags to False (optimization)
        rgb_frame.flags.writeable = False

        # Make detection predictions
        results = pose.process(rgb_frame)

        # Set frame memory back to True
        rgb_frame.flags.writeable = True

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
            frame, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )

        # Display the frame
        out.write(frame)
        cv2.imshow('Workout Window', frame)

        # Check for 'q' key press
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            print('Stopping live feed.')
            break

# Clean up
capture.release()
cv2.destroyAllWindows()
print("Program ended.")


