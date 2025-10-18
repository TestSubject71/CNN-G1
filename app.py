# Import the necessary libraries
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd

# Konfigurasi halaman
st.set_page_config(page_title="CNN Image Classifier", layout="centered")
st.title("Klasifikasi Gambar Menggunakan CNN")
st.write("Upload gambar untuk melihat hasil prediksi dari model CNN kamu.")

# Fungsi untuk memuat model (gunakan cache agar tidak reload setiap kali)
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

# Upload file
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    
    
    st.image(image, caption="Gambar yang diunggah", use_column_width=True)

    # BUTTON START PREDICTION
    if st.button("Prediksi Gambar"):
        with st.spinner("Model sedang menganalisis..."):
            

            # PREDICTION
            prediction = predict(image)
            pred_label = np.argmax(prediction)
            prediction_array = prediction[0]
            class_names = [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
            
            st.success(f"### 🟩 Prediksi: **{class_names[pred_label]}**")
            st.write(f"Confidence: `{np.max(prediction)*100:.2f}%`")
            
            # EVALUASI ---------
            top_prob = np.max(prediction_array)
            top_label_index = np.argmax(prediction_array)
            top_label_name = class_names[top_label_index]

            # List probability
            st.write("Semua Probabilitas:")
            st.dataframe({
                "Kelas": class_names,
                "Probabilitas": prediction[0]
            })

            CONFIDENCE_THRESHOLD = 0.50  # 50%
            if top_prob < CONFIDENCE_THRESHOLD:
                st.warning("⚠️ Model tidak terlalu yakin dengan prediksi ini.")
            
            st.markdown("---")

            # Probability 3 teratas
            st.subheader("Analisis 3 Prediksi Teratas:")

            top_n_indices = np.argsort(prediction_array)[-3:][::-1]
            
            cols = st.columns(3)
            for i, col in enumerate(cols):
                with col:
                    index = top_n_indices[i]
                    st.metric(
                        label=f"Pilihan #{i+1}: **{class_names[index]}**",
                        value=f"{prediction_array[index]*100:.2f}%"
                    )
            
            st.markdown("---")

            # Evaluasi Graph
            st.subheader("Distribusi Probabilitas (Semua Kelas):")
            
            prob_df = pd.DataFrame({
                "Kelas": class_names,
                "Probabilitas": prediction_array
            })
            
            prob_df = prob_df.sort_values(by="Probabilitas", ascending=False).reset_index(drop=True)
            
            st.bar_chart(prob_df.set_index("Kelas"))
            st.write("Data Lengkap:")
            st.dataframe(prob_df, hide_index=True)

   

# Footer
st.markdown("---")
st.caption("Made With Streamlit")

