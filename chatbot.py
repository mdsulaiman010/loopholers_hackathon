# import necessary libraries
import streamlit as st
import google.generativeai as genai
import os
import io
import time
from PIL import Image

# Title
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("Your Personal AI Chat")

# Configure your Gemini API key
genai.configure(api_key=os.getenv("GEMINI_API_KEY")) 

# ✅ Dropdown to choose model
available_models = {
    "Gemini 1.5 Flash (No video)": "gemini-1.5-flash",
    "Gemini 2.0 Flash (Video + Free Tier)": "gemini-2.0-flash",
    "Gemini 2.5 Flash (Video + Preview)": "gemini-2.5-flash",
    "Gemini 2.5 Pro (Full multimodal, paid)": "gemini-2.5-pro"
}

selected_model_label = st.selectbox("🔀 Choose Gemini Model", list(available_models.keys()), index=1)

# Display model name
model_name = available_models[selected_model_label]
st.caption(f"🔁 Using model: `{model_name}`")

# ⚠️ Warn if model does not support video
if "1.5" in model_name:
    st.warning("⚠️ This model does not support video input. You can still use images or text.")

# Initialize Gemini model
model = genai.GenerativeModel(model_name)

# Sidebar to select or create a new chat session
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}

session_name = st.sidebar.selectbox("Select a chat session", list(st.session_state.sessions.keys()) + ["Create New Session"])

# Create a new session
if session_name == "Create New Session":
    new_session_name = st.sidebar.text_input("Enter a name for the new session")
    if st.sidebar.button("Create Session"):
        if new_session_name:
            st.session_state.sessions[new_session_name] = []
            session_name = new_session_name
            st.sidebar.success(f"Session '{session_name}' created.")
        else:
            st.sidebar.error("Please enter a valid session name.")

# Set the active session
chat_history = st.session_state.sessions.get(session_name, [])

# Custom message bubble style
user_message_style = """
    <div style="background-color:#DCF8C6; border-radius:15px; padding:10px; margin-bottom:5px; width:fit-content;">
        <strong>You: </strong> 
        {message}
    </div>
"""
bot_message_style = """
    <div style="background-color:#E8E8E8; border-radius:15px; padding:10px; margin-bottom:5px; width:fit-content;">
        <strong>Bot: </strong>
        {message}
    </div>
"""

# Display chat history for the selected session
for msg in chat_history:
    role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
    st.markdown(f"**{role}:** {msg['text']}")

# Upload Section (Image or Video)
st.divider()
uploaded_file = st.file_uploader("📎 Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
media_prompt = st.text_input("Optional prompt to guide Gemini (e.g. 'Summarize this video')", key="media_prompt")

if uploaded_file is not None:
    file_type = uploaded_file.type
    st.markdown(f"✅ File uploaded: `{uploaded_file.name}`")

    # Handle image files
    if "image" in file_type:
        try:
            img = Image.open(uploaded_file)
            st.image(img, caption="📷 Uploaded Image", use_column_width=True)
            prompt = media_prompt if media_prompt else "Summarize the content of this image."
            response = model.generate_content([img, prompt])
            chat_history.append({"role": "user", "text": f"[Image] {prompt}"})
            chat_history.append({"role": "bot", "text": response.text})
        except Exception as e:
            st.error(f"Failed to analyze image: {e}")

    # Handle video files (requires video-capable model)
    if "video" in file_type:
        try:
            st.video(uploaded_file)

            # Upload file with correct MIME type
            blob = genai.upload_file(uploaded_file, mime_type="video/mp4")

            # add spinner while waiting for file to be ACTIVE
            with st.spinner("Processing uploaded video..."):
                # Wait for file to be ACTIVE (poll every 0.5s, max 10s)
                for _ in range(45):
                    file_status = genai.get_file(blob.name)
                    if str(file_status.state) == "State.ACTIVE":
                        st.success("✅ File is ACTIVE!")
                        break
                    time.sleep(0.5)
                else:
                    raise Exception("Uploaded video file is not ACTIVE after waiting.")

            # Prompt and summarize
            prompt = media_prompt if media_prompt else "Summarize this video."
            response = model.generate_content([blob, prompt])

            chat_history.append({"role": "user", "text": f"[Video] {prompt}"})
            chat_history.append({"role": "bot", "text": response.text})

            # ✅ Force rerun to refresh chat display
            st.rerun()

        except Exception as e:
            st.error(f"Failed to analyze video: {e}")

# Text Input handler
def handle_input():
    user_input = st.session_state.chat_input.strip()
    if user_input == "":
        return

    chat_history.append({"role": "user", "text": user_input})

    try:
        response = model.generate_content(user_input)
        bot_reply = response.text.replace("**", "").replace("*", "")
    except Exception as e:
        bot_reply = f"❌ Error: {e}"

    chat_history.append({"role": "bot", "text": bot_reply})
    st.session_state.chat_input = ""  # Clear input

# Update session state with the latest chat history
st.session_state.sessions[session_name] = chat_history

# User input widget
st.text_input("Ask me something...", key="chat_input", on_change=handle_input)
