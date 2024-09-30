import streamlit as st
import requests
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import uvicorn
from threading import Thread

# Spotify API Setup
SPOTIPY_CLIENT_ID = 'your spotify id '
SPOTIPY_CLIENT_SECRET = 'your spotify code'
auth_manager = SpotifyClientCredentials(SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET)
sp = spotipy.Spotify(auth_manager=auth_manager)

# Load CLIP Model for image processing
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# FastAPI backend for image analysis
app = FastAPI()

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # Analyze the image using CLIP model
    mood = analyze_image(image)

    # Fetch a music track based on the mood
    track_url = get_music_by_mood(mood)

    return JSONResponse(content={"music_url": track_url})

def analyze_image(image):
    # Prepare the inputs for the CLIP model
    text_inputs = [
        "a happy scene",
        "a calm scene",
        "an energetic scene",
        "a sad scene",
        "a relaxing scene",
        "angry scene",
        "bird chirping"
    ]
    inputs = clip_processor(text=text_inputs, images=image, return_tensors="pt", padding=True)

    # Perform the forward pass
    outputs = clip_model(**inputs)
    
    # Compare image and text embeddings (calculate similarity)
    logits_per_image = outputs.logits_per_image  # Image-text similarity scores
    probs = logits_per_image.softmax(dim=1)      # Softmax to get probabilities

    # Map the most probable text to a mood
    mood_index = probs.argmax().item()
    moods = [
        "happy",
        "calm",
        "energetic",
        "sad",
        "relaxing",
        "angry",
        "bird chirping"
    ]
    
    return moods[mood_index]

def get_music_by_mood(mood):
    # Search for playlists based on mood on Spotify
    if mood == "happy":
        result = sp.search(q='happy', type='playlist', limit=1)
    elif mood == "calm":
        result = sp.search(q='calm', type='playlist', limit=1)
    elif mood == "energetic":
        result = sp.search(q='energetic', type='playlist', limit=1)
    elif mood == "sad":
        result = sp.search(q='sad', type='playlist', limit=1)
    elif mood == "relaxing":
        result = sp.search(q='relaxing', type='playlist', limit=1)
    elif mood == "angry":
        result = sp.search(q='angry', type='playlist', limit=1)
    elif mood == "bird chirping":
        result = sp.search(q='bird chirping', type='playlist', limit=1)
    else:
        result = sp.search(q='chill', type='playlist', limit=1)

    # Fetch the playlist URL
    playlist_url = result['playlists']['items'][0]['external_urls']['spotify']
    return playlist_url


# Run FastAPI server in a separate thread
def run_fastapi():
    uvicorn.run(app, host="127.0.0.1", port=8053)

api_thread = Thread(target=run_fastapi, daemon=True)
api_thread.start()
page_bg_img = '''
<style>
.stApp {
  background-image: linear-gradient(to right, #ffccff, #ffffff);
  background-size: cover;
  color: black;
}

h1,h2,h3,h6,h4,h5{
  color: #4B0082;
}
p{
    color: #cea3ff;
}

.uploadedImage {
    border: 3px solid #4B0082;
    border-radius: 10px;
}

button.css-1q8dd3e {
    background-color: #ffffff;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 5px;
    font-weight: bold;
    font-size: 16px;
}

button.css-1q8dd3e:hover {
    background-color: #ffffff;
}
</style>
'''

# Apply the custom CSS
st.markdown(page_bg_img, unsafe_allow_html=True)


# Streamlit frontend UI
st.title('🎵 Image to Music Generator 🎶')
st.write('📷 Upload an image, and we’ll match it with appropriate background music! 🎧')

# Upload image
uploaded_image = st.file_uploader("Choose an image... 🖼️", type=["jpg", "jpeg", "png"])

# Once the image is uploaded
if uploaded_image is not None:
    image = Image.open(uploaded_image)

    # Show the uploaded image
    st.image(image, caption='Uploaded Image', use_column_width=True, output_format="PNG")
    
    # Convert image to bytes
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Make a request to the FastAPI backend
    if st.button('🎶 Get Music 🎵'):
        with st.spinner('Analyzing the image and fetching music...'):
            files = {'file': ('image.png', img_byte_arr, 'image/png')}
            response = requests.post("http://127.0.0.1:8053/upload-image", files=files)

            if response.status_code == 200:
                data = response.json()
                st.success('🎉 Music found!')

                # Display Spotify embed link by converting to embeddable format
                music_url = data.get('music_url').replace("open.spotify.com", "open.spotify.com/embed")
                st.markdown(f'<iframe src="{music_url}" width="500" height="380" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>', unsafe_allow_html=True)
            else:
                st.error('❌ Failed to get music. Please try again.')

# Credits section
def credits():
    st.markdown("### Credits")
    st.caption('''
        Made by batch -31 mini Project\n
        21H51A0592 21H51A0593 21H51A0594 
    ''')

# Display credits at the bottom
credits()
