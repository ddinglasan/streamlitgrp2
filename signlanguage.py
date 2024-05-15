import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import base64

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('signlanguage.h5')
    return model

model = load_model()

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to your image file
image_path = 'aslbg.png'

# Generate the base64 image
base64_image = get_base64_image(image_path)

# CSS to set the background image and adjust file uploader button
st.markdown(
    f"""
    <style>
    .stApp 
    {{
        background: url(data:image/png;base64,{base64_image}) no-repeat center center fixed;
        background-size: cover;
    }}
    .st-emotion-cache-13ln4jf 
    {{
        width: 100%;
        padding: 40rem 1rem 10rem;
        max-width: 46rem;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

file = st.file_uploader("Choose a hand gesture from the photos", type=["jpg", "png"])

def import_and_predict(image_data, model):
    size = (50, 50)  # Match the input size with the Google Colab code
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Use Image.LANCZOS for resizing
    img = np.asarray(image)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)

    # Reshape input according to the model's input shape
    img_reshape = tf.image.resize(img, [64, 64])  # Resize to (64, 64)
    
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8',
                   '9', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h',
                   'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
                   'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    string = "OUTPUT : " + class_names[np.argmax(prediction)]
    st.success(string)