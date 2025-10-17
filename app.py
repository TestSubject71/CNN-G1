# Import the necessary libraries
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Page config
st.set_page_config(page_title="Image Classifier", page_icon="🖼️")

@st.cache_resource
def load_our_model():
    model = load_model('cifar10_cnn_model.keras')
    return model

model = load_our_model()

# Function Prediksi from image upload
def predict(image):
   
    image = image.resize((32, 32))
    image_array = np.array(image)

    if image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    image_array = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_array)
    prediction = model.predict(processed_image)

    return prediction

