# Import the necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="CNN Image Classifier", layout="centered")
st.title("🧠 Klasifikasi Gambar Menggunakan CNN")
st.write("Upload gambar untuk melihat hasil prediksi dari model CNN kamu.")

# Fungsi untuk memuat model (gunakan cache agar tidak reload setiap kali)
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("model_cnn.h5")
    return model

# Panggil model
model = load_cnn_model()

# Upload file
uploaded_file = st.file_uploader("📤 Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)
    st.write("⏳ Sedang memproses...")

    # Preprocessing sesuai kebutuhan model
    img = image.resize((150, 150))  # sesuaikan ukuran dengan input model kamu
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prediksi
    prediction = model.predict(img_array)
    pred_label = np.argmax(prediction, axis=1)[0]

    # Daftar label (ubah sesuai label dataset kamu)
    class_names = ['Cat', 'Dog', 'Horse', 'Elephant']

    st.success(f"### 🟩 Prediksi: **{class_names[pred_label]}**")
    st.write(f"Confidence: `{np.max(prediction)*100:.2f}%`")

# Footer
st.markdown("---")
st.caption("Dibuat dengan ❤️ menggunakan Streamlit")

