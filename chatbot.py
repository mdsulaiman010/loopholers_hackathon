# import necessary libraries
import streamlit as st
import google.generativeai as genai
import os
import io
import time
import pandas as pd
from PIL import Image
from datetime import datetime

# Title
st.set_page_config(page_title="AI Chatbot", layout="centered")
st.title("Your Personal Fitness AI Chat")
st.markdown("""
    <style>
        /* Custom fonts and background */
        body { background-color: #f7fafc; }
        .main { background-color: #fff; border-radius: 16px; padding: 24px; box-shadow: 0 0 20px #e4e4e4; }
        /* Chat bubbles */
        .user-bubble {
            background: #0074D9;
            color: white;
            border-radius: 18px 18px 4px 18px;
            padding: 10px 16px;
            margin: 6px 0 6px 40px;
            max-width: 80%;
            align-self: flex-end;
        }
        .bot-bubble {
            background: #e2e8f0;
            color: #222;
            border-radius: 18px 18px 18px 4px;
            padding: 10px 16px;
            margin: 6px 40px 6px 0;
            max-width: 80%;
            align-self: flex-start;
        }
        .sidebar-user {
            text-align:center;
            margin-top:24px;
        }
        .sidebar-user img {
            border-radius:50%;
            width:80px;
            margin-bottom:8px;
        }
        .sidebar-user h4 {
            margin:0;
        }
    </style>
""", unsafe_allow_html=True)

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
with st.sidebar:
    st.markdown("""
    <div class="sidebar-user">
        <img src="https://ui-avatars.com/api/?name=AI+User&background=0074D9&color=fff" />
        <h4>Welcome!</h4>
        <p style='font-size:13px'>Your sessions are listed below.</p>
    </div>
    """, unsafe_allow_html=True)
    
if session_name != "Create New Session" and st.sidebar.button("🗑️ Delete This Session"):
    del st.session_state.sessions[session_name]
    st.sidebar.success(f"Session '{session_name}' deleted.")
    st.experimental_rerun()


# # Toggle button to switch to Webcam UI
if st.button("Switch to Live AI Analysis"):
    st.switch_page("webcam_ui_bharathaan.py")

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
# for msg in chat_history:
#     role = "🧑 You" if msg["role"] == "user" else "🤖 Bot"
#     st.markdown(f"**{role}:** {msg['text']}")
st.markdown("<div style='display:flex; flex-direction:column;'>", unsafe_allow_html=True)
for msg in chat_history:
    time = msg.get("time", "")
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>{msg['text']} <span style='font-size:11px;float:right;'>{time}</span></div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>{msg['text']} <span style='font-size:11px;float:right;'>{time}</span></div>", unsafe_allow_html=True)

# Upload Section (Image or Video)
st.divider()
# uploaded_file = st.file_uploader("📎 Upload an image or video", type=["jpg", "jpeg", "png", "mp4"])
# media_prompt = st.text_input("Optional prompt to guide Gemini (e.g. 'Summarize this video')", key="media_prompt")
col1, col2 = st.columns([2, 3])
with col1:
    uploaded_file = st.file_uploader("📎 Upload image/video", type=["jpg", "jpeg", "png", "mp4"])
with col2:
    media_prompt = st.text_input("Prompt for uploaded media", key="media_prompt", placeholder="E.g. 'Summarize this video'")

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
    
    timestamp = datetime.now().strftime("%H:%M")
    chat_history.append({"role": "user", "text": user_input, "time": timestamp})

    try:
        response = model.generate_content(user_input)
        bot_reply = response.text.replace("**", "").replace("*", "")
    except Exception as e:
        bot_reply = f"❌ Error: {e}"

    chat_history.append({"role": "bot", "text": bot_reply, "time": timestamp})
    st.session_state.chat_input = ""  # Clear input

# Update session state with the latest chat history
st.session_state.sessions[session_name] = chat_history

if len(chat_history) > 0:
    chat_text = "\n".join(
        [f"You: {msg['text']}" if msg["role"] == "user" else f"Bot: {msg['text']}" for msg in chat_history]
    )
    st.download_button("💾 Download Chat", chat_text, file_name=f"{session_name}.txt")

# User input widget
st.markdown("---")
st.text_input("Type your message and press Enter", key="chat_input", on_change=handle_input, placeholder="Ask me anything...", label_visibility="collapsed")

