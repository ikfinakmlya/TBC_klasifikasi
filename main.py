import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load model (.keras)
@st.cache_resource  # biar model cuma diload sekali
def load_model():
    return tf.keras.models.load_model("cnn_tbc_model.keras")

model = load_model()

# Judul Aplikasi
st.title("Klasifikasi Citra TBC vs Normal")
st.write("Upload gambar X-ray untuk deteksi **TBC** menggunakan model CNN.")

# Upload gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Preprocessing gambar
    img = image.load_img(uploaded_file, target_size=(64,64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Prediksi
    prediction = model.predict(img_array)
    label = "TBC" if prediction[0][0] > 0.5 else "Normal"

    # Output
    st.image(img, caption=f"Hasil Prediksi: {label}", use_container_width=True)
    st.success(f"✅ Prediksi Model: **{label}**")
