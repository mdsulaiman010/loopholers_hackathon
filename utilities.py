import google.generativeai as genai
import cv2
import numpy as np
from math import degrees, acos

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
        print(f"Uploading video: {file_path} with display name: {display_name}")
        video_file = genai.upload_file(path=file_path, display_name=display_name)
        print(f"Uploaded video file: {video_file.name}")
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
        print(f"Generating content for video: {video_file.name}")
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
    return np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

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
    Process left arm bicep curl, drawing zones and returning key points.
    
    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h, w: Frame height and width.
    
    Returns:
        Dictionary with wrist coordinates and zone centers.
    """
    left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
    
    left_top_zone = [left_shoulder[0] * w, left_shoulder[1] * h - 50]
    left_bottom_zone = [left_shoulder[0] * w, left_shoulder[1] * h + 100]
    
    cv2.circle(frame_rgb, (int(left_top_zone[0]), int(left_top_zone[1])), 25, (0, 255, 0), 2)
    cv2.circle(frame_rgb, (int(left_bottom_zone[0]), int(left_bottom_zone[1])), 25, (255, 0, 0), 2)
    
    return {
        'left_wrist': [left_wrist[0] * w, left_wrist[1] * h],
        'left_top_zone': left_top_zone,
        'left_bottom_zone': left_bottom_zone
    }

def right_arm_bicep_curl(landmarks, frame_rgb, h, w):
    """
    Process right arm bicep curl, drawing zones and returning key points.
    
    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h, w: Frame height and width.
    
    Returns:
        Dictionary with wrist coordinates and zone centers.
    """
    right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]
    right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    
    right_top_zone = [right_shoulder[0] * w, right_shoulder[1] * h - 50]
    right_bottom_zone = [right_shoulder[0] * w, right_shoulder[1] * h + 100]
    
    cv2.circle(frame_rgb, (int(right_top_zone[0]), int(right_top_zone[1])), 25, (0, 255, 0), 2)
    cv2.circle(frame_rgb, (int(right_bottom_zone[0]), int(right_bottom_zone[1])), 25, (255, 0, 0), 2)
    
    return {
        'right_wrist': [right_wrist[0] * w, right_wrist[1] * h],
        'right_top_zone': right_top_zone,
        'right_bottom_zone': right_bottom_zone
    }

def both_arms_bicep_curl(landmarks, frame_rgb, h, w):
    """
    Process both arms bicep curl, drawing zones and returning key points.
    
    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h, w: Frame height and width.
    
    Returns:
        Dictionary with wrist coordinates and zone centers.
    """
    left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHoulder.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    
    left_top_zone = [left_shoulder[0] * w, left_shoulder[1] * h - 50]
    left_bottom_zone = [left_shoulder[0] * w, left_shoulder[1] * h + 100]
    right_top_zone = [right_shoulder[0] * w, right_shoulder[1] * h - 50]
    right_bottom_zone = [right_shoulder[0] * w, right_shoulder[1] * h + 100]
    
    cv2.circle(frame_rgb, (int(left_top_zone[0]), int(left_top_zone[1])), 25, (0, 255, 0), 2)
    cv2.circle(frame_rgb, (int(left_bottom_zone[0]), int(left_bottom_zone[1])), 25, (255, 0, 0), 2)
    cv2.circle(frame_rgb, (int(right_top_zone[0]), int(right_top_zone[1])), 25, (0, 255, 0), 2)
    cv2.circle(frame_rgb, (int(right_bottom_zone[0]), int(right_bottom_zone[1])), 25, (255, 0, 0), 2)
    
    return {
        'left_wrist': [left_wrist[0] * w, left_wrist[1] * h],
        'right_wrist': [right_wrist[0] * w, right_wrist[1] * h],
        'left_top_zone': left_top_zone,
        'left_bottom_zone': left_bottom_zone,
        'right_top_zone': right_top_zone,
        'right_bottom_zone': right_bottom_zone
    }

def both_arms_lateral_raise(landmarks, frame_rgb, h, w):
    """
    Process both arms lateral raise, drawing zones and returning key points.
    
    Args:
        landmarks: Mediapipe pose landmarks.
        frame_rgb: RGB frame for drawing.
        h, w: Frame height and width.
    
    Returns:
        Dictionary with wrist coordinates and zone centers.
    """
    left_wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value].y]
    left_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    
    left_top_zone = [left_shoulder[0] * w - 100, left_shoulder[1] * h]
    left_bottom_zone = [left_shoulder[0] * w, left_shoulder[1] * h + 100]
    right_top_zone = [right_shoulder[0] * w + 100, right_shoulder[1] * h]
    right_bottom_zone = [right_shoulder[0] * w, right_shoulder[1] * h + 100]
    
    cv2.circle(frame_rgb, (int(left_top_zone[0]), int(left_top_zone[1])), 25, (0, 255, 0), 2)
    cv2.circle(frame_rgb, (int(left_bottom_zone[0]), int(left_bottom_zone[1])), 25, (255, 0, 0), 2)
    cv2.circle(frame_rgb, (int(right_top_zone[0]), int(right_top_zone[1])), 25, (0, 255, 0), 2)
    cv2.circle(frame_rgb, (int(right_bottom_zone[0]), int(right_bottom_zone[1])), 25, (255, 0, 0), 2)
    
    return {
        'left_wrist': [left_wrist[0] * w, left_wrist[1] * h],
        'right_wrist': [right_wrist[0] * w, right_wrist[1] * h],
        'left_top_zone': left_top_zone,
        'left_bottom_zone': left_bottom_zone,
        'right_top_zone': right_top_zone,
        'right_bottom_zone': right_bottom_zone
    }