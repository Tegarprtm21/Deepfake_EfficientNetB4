import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import gdown
from mtcnn.mtcnn import MTCNN
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Deepfake", page_icon="🧠", layout="centered")

# Fungsi untuk memuat model dari Google Drive jika belum ada
@st.cache_resource
def load_model():
    model_path = "model/final_model.h5"
    
    if not os.path.exists(model_path):
        st.info("📥 Model belum ditemukan. Mengunduh dari Google Drive...")
        os.makedirs("model", exist_ok=True)

        # ID dari model Google Drive kamu
        file_id = "1jgfFQrm0HkjDbGOpU6rb_7rEfZ1xzVZC"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        gdown.download(url, model_path, quiet=False)

    return tf.keras.models.load_model(model_path, compile=False)

model = load_model()

# Fungsi untuk memuat detektor wajah
@st.cache_resource
def load_detector():
    return MTCNN()

detector = load_detector()

# Ekstraksi wajah dari gambar input
def extract_face(image_np, padding=20, target_size=380):
    faces = detector.detect_faces(image_np)
    if not faces:
        return None
    face = sorted(faces, key=lambda x: x['confidence'], reverse=True)[0]
    x, y, w, h = face['box']
    x, y = max(0, x - padding), max(0, y - padding)
    w, h = w + 2 * padding, h + 2 * padding
    h_img, w_img, _ = image_np.shape
    w = min(w, w_img - x)
    h = min(h, h_img - y)
    face_crop = image_np[y:y+h, x:x+w]
    resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
    return resized

# Sidebar navigasi
st.sidebar.title("🔍 Navigasi")
menu = st.sidebar.radio("📂 Menu", ["🏠 Beranda", "🧠 Deteksi Deepfake"])

# Halaman Beranda
if menu == "🏠 Beranda":
    st.markdown("<h1 style='text-align: center;'>📌 Deteksi Gambar Deepfake</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="text-align: justify; font-size: 17px;">
        Selamat datang di aplikasi <b>Deteksi Gambar Deepfake</b> berbasis <b>EfficientNet-B4</b> dan <b>Streamlit</b>!

        Aplikasi ini dirancang untuk mendeteksi gambar wajah yang telah dimanipulasi menggunakan teknik <b>deepfake</b>. Sistem ini memanfaatkan teknologi <b>deep learning</b> dan melalui beberapa tahapan utama:

        <ul>
        <li>📤 Pengguna mengunggah gambar wajah</li>
        <li>🔍 Deteksi wajah dilakukan dengan <b>MTCNN</b> untuk mengambil area wajah</li>
        <li>✂️ Gambar wajah dicrop dan diubah ukurannya menjadi 380x380 piksel</li>
        <li>🤖 Model <b>EfficientNet-B4</b> akan memproses gambar dan memberikan prediksi</li>
        </ul>

        Anda akan mendapatkan hasil prediksi apakah wajah tersebut termasuk <b>REAL (Asli)</b> atau <b>FAKE (Palsu)</b> disertai dengan tingkat probabilitas keyakinan model.

        <br>
        👉 Untuk memulai, klik menu <b>🧠 Deteksi Deepfake</b> di sebelah kiri!
        </div>
        """,
        unsafe_allow_html=True
    )

# Halaman Deteksi
elif menu == "🧠 Deteksi Deepfake":
    st.markdown("<h1 style='text-align: center;'>🧠 Deteksi Gambar Deepfake</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Unggah gambar wajah (.jpg/.jpeg/.png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        if image_np.shape[0] < 100 or image_np.shape[1] < 100:
            st.warning("⚠️ Gambar terlalu kecil. Gunakan gambar dengan resolusi lebih tinggi.")
        else:
            st.markdown("### 📸 Ini adalah gambar yang Anda unggah:")
            st.image(image_np, use_column_width=True)

            st.markdown("### 🔍 Mendeteksi wajah menggunakan MTCNN...")
            face_img = extract_face(image_np)

            if face_img is not None:
                st.success("✅ Wajah berhasil terdeteksi.")
                st.markdown("### ✂️ Berikut adalah hasil crop dan resize wajah:")
                st.image(face_img, width=300)

                st.markdown("### 🤖 Melakukan prediksi dengan model EfficientNet-B4...")
                with st.spinner("⏳ Sedang memproses prediksi..."):
                    input_array = preprocess_input(face_img.astype(np.float32))
                    input_array = np.expand_dims(input_array, axis=0)
                    prediction = model.predict(input_array)[0][0]
                    label = "🟢 REAL (Asli)" if prediction >= 0.5 else "🔴 FAKE (Palsu)"
                    confidence = prediction if label.startswith("🟢") else 1 - prediction

                st.markdown(f"<h2 style='color:#117A65;'>Hasil Deteksi: {label}</h2>", unsafe_allow_html=True)
                st.markdown(f"<h4>🔢 Probabilitas Prediksi: <b>{confidence * 100:.2f}%</b></h4>", unsafe_allow_html=True)
                st.caption("Prediksi dilakukan berdasarkan ambang batas 0.5")
            else:
                st.error("🚫 Wajah tidak terdeteksi. Pastikan gambar mengandung wajah yang jelas dan menghadap ke kamera.")
