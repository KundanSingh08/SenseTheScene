import cv2
import streamlit as st
import tempfile
import os
import base64
import numpy as np
from groq import Groq
from gtts import gTTS

# ------------------ Configuration ------------------
vision_model = "llama-3.2-11b-vision-preview"
groq_api_key = "your_groq_api_key_here"
language_model  = "llama-3.1-8b-instant"
groq_client = Groq(api_key="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")

# ------------------ Helper Functions ------------------
def get_image_size(image):
    """Returns the size of the image in bytes."""
    success, img_encoded = cv2.imencode(".png", image)
    if success:
        return len(img_encoded)
    return 0

def create_frame_grid(frames, max_size=4 * 1024 * 1024):
    """Creates a 2x3 grid from six frames and checks its size."""
    if len(frames) < 6:
        print("Not enough frames to create grid")
        return None
    try:
        row1 = np.hstack(frames[:3])
        row2 = np.hstack(frames[3:])
        grid = np.vstack([row1, row2])


        return grid
    except Exception as e:
        print(f"Error creating frame grid: {e}")
        return None

def query_vision(image):
    try:
        success, img_encoded = cv2.imencode(".png", image)
        if not success:
            print("Image encoding failed")
            return "Image encoding failed"
        
        b64_data = base64.b64encode(img_encoded).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_data}"
        messages = [
            {"role": "user", "content": "Generate a descriptive caption for this scene grid of 6 frames from a single video in a way that conveys the scene to a visually impaired person under 10-15 words so they can imagine it."},
            {"role": "user", "content": [
                {"type": "text", "text": ""},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]}
        ]
        
        completion = groq_client.chat.completions.create(
            model=vision_model,
            messages=messages,
            temperature=.5,
            max_completion_tokens=100,
            top_p=1,
            stream=False,
        )
        
        if completion.choices:
            return completion.choices[0].message.content.strip()
        return "No caption generated"
    except Exception as e:
        print(f"Error querying Groq API: {e}")
        return None

def generate_tts(prompt):
    try:
        tts = gTTS(prompt, lang='en')
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        tts.save(audio_path)
        return audio_path
    except Exception as e:
        print(f"Error generating TTS: {e}")
        return None

def play_video_with_audio(video_path, audio_path, container):
    try:
        file_extension = video_path.split(".")[-1].lower()
        # Ensure MOV is treated as MP4 for better browser compatibility
        mime_type = "video/mp4" if file_extension in ["mp4", "mov"] else "video/quicktime"

        with open(video_path, 'rb') as vid_file:
            video_bytes = vid_file.read()
            video_base64 = base64.b64encode(video_bytes).decode('utf-8')

        with open(audio_path, 'rb') as aud_file:
            audio_bytes = aud_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        video_html = f'''
        <video id="video" width="640" height="360" controls autoplay>
            <source src="data:{mime_type};base64,{video_base64}" type="{mime_type}">
            Your browser does not support the video tag.
        </video>
        <audio id="audio" autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            Your browser does not support the audio tag.
        </audio>
        <script>
          document.getElementById('video').addEventListener('play', function() {{
              document.getElementById('audio').play();
          }});
        </script>
        '''
        container.markdown(video_html, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error playing video with audio: {e}")

def summarize_captions(captions, length=50):
    try:
        prompt = f"Summarize the following descriptions from sequential clips from a single video under {length} words as the only output" + " ".join(captions)
        completion = groq_client.chat.completions.create(
            model=language_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=.3,
            max_completion_tokens=200,
            top_p=1,
            stream=False,
        )
        if completion.choices:
            return completion.choices[0].message.content.strip()
        return "No final caption generated"
    except Exception as e:
        print(f"Error summarizing captions: {e}")
        return None


# ------------------ Streamlit UI ------------------
st.title("🎥 Live Video Captioning")
st.write("Upload a video, and a final detailed caption will be generated.")

video_placeholder = st.empty()
uploaded_video = st.file_uploader("Upload your video (MP4/MOV)", type=["mp4", "mov"])

if uploaded_video:
    if "last_uploaded_video" not in st.session_state or st.session_state.last_uploaded_video != uploaded_video.name:
        st.session_state.last_uploaded_video = uploaded_video.name
        st.session_state.video_processed = False
        video_placeholder.empty()

final_caption_placeholder = st.empty()

if uploaded_video and not st.session_state.video_processed:
    try:
        file_extension = uploaded_video.name.split(".")[-1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_video:
            temp_video.write(uploaded_video.read())
            temp_video_path = temp_video.name
        
        cap = cv2.VideoCapture(temp_video_path)
        if not cap.isOpened():
            print("Error: Could not open video file")
        
        all_captions = []
        frame_count = 0
        frame_skip = 15
        frame_buffer = []
        
        progress_bar = st.progress(0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_skip == 0:
                resized_frame = cv2.resize(frame, (800, 600))  # Resize to 800x600
                frame_buffer.append(resized_frame)
                if len(frame_buffer) == 6:
                    grid = create_frame_grid(frame_buffer)
                    if grid is not None:
                        caption = query_vision(grid)
                        if caption:
                            all_captions.append(caption)
                    frame_buffer.clear()
            
            progress_bar.progress(min(100, int((frame_count / max(1, total_frames)) * 100)))
            frame_count += 1
        cap.release()
        
        if all_captions:
            final_caption = summarize_captions(all_captions, max(1, (total_frames // 30) * 2))
            final_caption_placeholder.markdown(f"## Final Caption: {final_caption}")
            audio_path = generate_tts(final_caption)
            play_video_with_audio(temp_video_path, audio_path, video_placeholder)
        
        st.session_state.video_processed = True
    except Exception as e:
        print(f"Error processing video: {e}")


      