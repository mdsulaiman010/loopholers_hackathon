import google.generativeai as genai
import cv2
import numpy as np
from math import degrees, acos
import mediapipe as mp


def upload_and_process_video(file_path, display_name):
    """
    Uploads a video file to the Gemini API and returns the processed file object.

    Args:
        file_path (str): Path to the video file to upload.
        display_name (str): Display name for the uploaded file.

    Returns:
        File object if successful, None otherwise.
    """
    try:
        # print(f"Uploading video: {file_path} with display name: {display_name}")
        video_file = genai.upload_file(path=file_path, display_name=display_name)
        # print(f"Uploaded video file: {video_file.name}")
        return video_file
    except Exception as e:
        print(f"Error uploading video: {e}")
        return None


def generate_content_from_video(video_file, prompt):
    """
    Generates content from a video file using the Gemini model.

    Args:
        video_file: Uploaded video file object.
        prompt (str): Prompt to guide content generation.

    Returns:
        Generated content as a string, or None if failed.
    """
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        # print(f"Generating content for video: {video_file.name}")
        response = model.generate_content([video_file, prompt])
        content = response.text.strip()
        print(f"Generated content: {content}")
        return content
    except Exception as e:
        print(f"Error generating content: {e}")
        return None


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points a, b, c in degrees.

    Args:
        a, b, c: Lists of [x, y] coordinates.

    Returns:
        Angle in degrees.
    """
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = degrees(acos(cosine_angle))
    return angle


def calculate_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.

    Args:
        point1, point2: Lists of [x, y] coordinates.

    Returns:
        Distance as a float.
    """
    return np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def is_inside_zone(point, zone_center, radius=30):
    """
    Check if a point is inside a circular zone.

    Args:
        point: List of [x, y] coordinates.
        zone_center: List of [x, y] coordinates for the zone center.
        radius: Radius of the zone in pixels (default: 30).

    Returns:
        Boolean indicating if the point is inside the zone.
    """
    return calculate_distance(point, zone_center) <= radius


def left_arm_bicep_curl(landmarks, frame_rgb, h, w):
    """
    Process left arm bicep curl, drawing zones and wrist point, and returning key points.

    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h (int): Frame height.
        w (int): Frame width.

    Returns:
        dict: Dictionary with wrist coordinates and zone centers for left arm, with placeholders for right arm.
    """
    try:
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]

        shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        hip_coords = (int(left_hip.x * w), int(left_hip.y * h))
        wrist_coords = (int(left_wrist.x * w), int(left_wrist.y * h))

        top_target = (shoulder_coords[0], shoulder_coords[1] + 50)
        bottom_target = (hip_coords[0] + 30, hip_coords[1] - 10)

        cv2.circle(frame_rgb, top_target, 25, (0, 255, 0), 2)
        cv2.circle(frame_rgb, bottom_target, 25, (0, 0, 255), 2)
        cv2.circle(frame_rgb, wrist_coords, 8, (255, 255, 255), -1)

        return {
            "left_wrist": wrist_coords,
            "right_wrist": None,
            "left_top_zone": top_target,
            "right_top_zone": None,
            "left_bottom_zone": bottom_target,
            "right_bottom_zone": None,
        }
    except (IndexError, AttributeError) as e:
        print(f"Error processing left arm bicep curl: {e}")
        return {
            "left_wrist": None,
            "right_wrist": None,
            "left_top_zone": None,
            "right_top_zone": None,
            "left_bottom_zone": None,
            "right_bottom_zone": None,
        }


def right_arm_bicep_curl(landmarks, frame_rgb, h, w):
    """
    Process right arm bicep curl, drawing zones and wrist point, and returning key points.

    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h (int): Frame height.
        w (int): Frame width.

    Returns:
        dict: Dictionary with wrist coordinates and zone centers for right arm, with placeholders for left arm.
    """
    try:
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

        shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        wrist_coords = (int(right_wrist.x * w), int(right_wrist.y * h))
        hip_coords = (int(right_hip.x * w), int(right_hip.y * h))

        top_target = (shoulder_coords[0] - 10, shoulder_coords[1] + 30)
        bottom_target = (hip_coords[0] - 50, hip_coords[1] - 10)

        cv2.circle(frame_rgb, top_target, 25, (0, 255, 0), 2)
        cv2.circle(frame_rgb, bottom_target, 25, (0, 0, 255), 2)
        cv2.circle(frame_rgb, wrist_coords, 8, (255, 255, 255), -1)

        return {
            "left_wrist": None,
            "right_wrist": wrist_coords,
            "left_top_zone": None,
            "right_top_zone": top_target,
            "left_bottom_zone": None,
            "right_bottom_zone": bottom_target,
        }
    except (IndexError, AttributeError) as e:
        print(f"Error processing right arm bicep curl: {e}")
        return {
            "left_wrist": None,
            "right_wrist": None,
            "left_top_zone": None,
            "right_top_zone": None,
            "left_bottom_zone": None,
            "right_bottom_zone": None,
        }


def both_arms_bicep_curl(landmarks, frame_rgb, h, w):
    """
    Process both arms bicep curl, drawing zones and wrist points, and returning key points.

    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h (int): Frame height.
        w (int): Frame width.

    Returns:
        dict: Dictionary with wrist coordinates and zone centers for both arms.
    """
    try:
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

        left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        left_wrist_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        right_wrist_coords = (int(right_wrist.x * w), int(right_wrist.y * h))

        left_top_zone = (left_shoulder_coords[0], left_shoulder_coords[1] + 50)
        left_bottom_zone = (left_shoulder_coords[0] + 30, left_shoulder_coords[1] + 150)
        right_top_zone = (right_shoulder_coords[0] - 10, right_shoulder_coords[1] + 30)
        right_bottom_zone = (
            right_shoulder_coords[0] - 50,
            right_shoulder_coords[1] + 130,
        )

        cv2.circle(frame_rgb, left_top_zone, 25, (0, 255, 0), 2)
        cv2.circle(frame_rgb, left_bottom_zone, 25, (0, 0, 255), 2)
        cv2.circle(frame_rgb, right_top_zone, 25, (0, 255, 0), 2)
        cv2.circle(frame_rgb, right_bottom_zone, 25, (0, 0, 255), 2)
        cv2.circle(frame_rgb, left_wrist_coords, 8, (255, 255, 255), -1)
        cv2.circle(frame_rgb, right_wrist_coords, 8, (255, 255, 255), -1)

        return {
            "left_wrist": left_wrist_coords,
            "right_wrist": right_wrist_coords,
            "left_top_zone": left_top_zone,
            "left_bottom_zone": left_bottom_zone,
            "right_top_zone": right_top_zone,
            "right_bottom_zone": right_bottom_zone,
        }
    except (IndexError, AttributeError) as e:
        print(f"Error processing both arms bicep curl: {e}")
        return {
            "left_wrist": None,
            "right_wrist": None,
            "left_top_zone": None,
            "left_bottom_zone": None,
            "right_top_zone": None,
            "right_bottom_zone": None,
        }


def both_arms_lateral_raise(landmarks, frame_rgb, h, w):
    """
    Process both arms lateral raise, drawing zones and webcam points, and returning key points.

    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h (int): Frame height.
        w (int): Frame width.

    Returns:
        dict: Dictionary with wrist coordinates and zone centers for both arms.
    """
    try:
        left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value]
        left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
        left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value]
        right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value]
        right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
        right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value]

        left_shoulder_coords = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        left_hip_coords = (int(left_hip.x * w), int(left_hip.y * h))
        left_wrist_coords = (int(left_wrist.x * w), int(left_wrist.y * h))
        right_shoulder_coords = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        right_hip_coords = (int(right_hip.x * w), int(right_hip.y * h))
        right_wrist_coords = (int(right_wrist.x * w), int(right_wrist.y * h))

        left_top_zone = (left_shoulder_coords[0] + 100, left_shoulder_coords[1] - 80)
        left_bottom_zone = (left_hip_coords[0] + 30, left_hip_coords[1] - 10)
        right_top_zone = (right_shoulder_coords[0] - 100, right_shoulder_coords[1] - 80)
        right_bottom_zone = (right_hip_coords[0] - 20, right_hip_coords[1] + 30)

        cv2.circle(frame_rgb, left_top_zone, 25, (0, 255, 0), 2)
        cv2.circle(frame_rgb, left_bottom_zone, 25, (0, 0, 255), 2)
        cv2.circle(frame_rgb, right_top_zone, 25, (0, 255, 0), 2)
        cv2.circle(frame_rgb, right_bottom_zone, 25, (0, 0, 255), 2)
        cv2.circle(frame_rgb, left_wrist_coords, 8, (255, 255, 255), -1)
        cv2.circle(frame_rgb, right_wrist_coords, 8, (255, 255, 255), -1)

        return {
            "left_wrist": left_wrist_coords,
            "right_wrist": right_wrist_coords,
            "left_top_zone": left_top_zone,
            "left_bottom_zone": left_bottom_zone,
            "right_top_zone": right_top_zone,
            "right_bottom_zone": right_bottom_zone,
        }
    except (IndexError, AttributeError) as e:
        print(f"Error processing both arms lateral raise: {e}")
        return {
            "left_wrist": None,
            "right_wrist": None,
            "left_top_zone": None,
            "left_bottom_zone": None,
            "right_top_zone": None,
            "right_bottom_zone": None,
        }
