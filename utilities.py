import google.generativeai as genai
import time
import numpy as np


def upload_and_process_video(file_path, display_name):
    print("Uploading video file...")
    try:
        video_file = genai.upload_file(path=file_path, display_name=display_name, mime_type="video/mp4")
        print(f"Completed upload: {video_file.uri}")

        # Check the state of the uploaded file
        while video_file.state.name == "PROCESSING":
            print(".", end="", flush=True)
            time.sleep(10)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise ValueError("Video processing failed.")

        return video_file
    except Exception as e:
        print(f"Error uploading video: {e}")
        return None
    
def generate_content_from_video(video_file, prompt):
    try:
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        print("\nMaking LLM inference request...")
        response = model.generate_content([video_file, prompt], request_options={"timeout": 600})
        return response.text
    except Exception as e:
        print(f"Error generating content: {e}")
        return None
    

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