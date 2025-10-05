import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# =============================
# Load model CNN kamu
# =============================
@st.cache_resource
def load_cnn_model():
    model = load_model('cnn_tbc_model.h5')
    return model

model = load_cnn_model()

# =============================
# Tampilan utama aplikasi
# =============================
st.set_page_config(page_title="Klasifikasi Paru-Paru TBC", page_icon="🫁", layout="centered")
st.title("🩺 Klasifikasi Paru-Paru TBC Menggunakan CNN (MobileNetV2)")
st.write("Upload gambar X-ray paru-paru untuk mengetahui apakah **TBC** atau **Normal.**")

uploaded_file = st.file_uploader("📤 Upload gambar (jpg, jpeg, png):", type=["jpg", "jpeg", "png"])

# =============================
# Fungsi prediksi
# =============================
def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]
    label = "TBC" if prediction > 0.5 else "Normal"
    confidence = prediction if label == "TBC" else 1 - prediction
    return label, float(confidence)

# =============================
# Proses upload & hasil prediksi
# =============================
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="🩻 Gambar yang diunggah", use_column_width=True)

    with st.spinner("Sedang memproses gambar..."):
        label, confidence = predict_image(img)

    st.success(f"**Prediksi: {label}**")
    st.write(f"Tingkat keyakinan model: **{confidence*100:.2f}%**")

    if label == "TBC":
        st.warning("⚠️ Hasil menunjukkan indikasi TBC. Konsultasikan dengan dokter untuk pemeriksaan lanjutan.")
    else:
        st.info("✅ Tidak terdeteksi tanda TBC pada gambar ini.")

st.markdown("---")
st.caption("Model: MobileNetV2 • Dibuat oleh: [Nama Kamu]")

