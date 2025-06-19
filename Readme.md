# 🧠 Deteksi Gambar Deepfake dengan EfficientNet-B4

Aplikasi ini merupakan sistem deteksi gambar wajah palsu (deepfake) berbasis deep learning yang dikembangkan dengan model **EfficientNet-B4** dan dijalankan melalui antarmuka interaktif menggunakan **Streamlit**.

## 📌 Deskripsi

Deepfake adalah teknik manipulasi wajah menggunakan kecerdasan buatan yang dapat menimbulkan penyebaran informasi palsu. Aplikasi ini bertujuan untuk membantu dalam mendeteksi gambar wajah yang telah dimanipulasi.

### Proses Deteksi:
- 📤 Unggah gambar wajah (format JPG/PNG)
- 🔍 Deteksi wajah menggunakan **MTCNN**
- ✂️ Crop dan resize gambar menjadi 380x380 piksel
- 🧠 Prediksi menggunakan model **EfficientNet-B4**
- 📊 Menampilkan hasil: **REAL (Asli)** atau **FAKE (Palsu)** beserta probabilitasnya

## 🚀 Demo Aplikasi

🔗 [Klik untuk mencoba aplikasi](https://YOUR-STREAMLIT-URL.streamlit.app/)

> `URL`

## 🛠️ Cara Menjalankan Secara Lokal

1. Clone repository ini:
   git clone https://github.com/Tegarprtm21/Deepfake_EfficientNetB4.git
   cd Deepfake_EfficientNetB4
2. Install dependencies:
   pip install -r requirements.txt
3. Jalankan aplikasi:
streamlit run app.py
> Pastikan file model `EfficientNetB4.h5` tersedia di folder `model/`.

## 📦 Dependencies
* TensorFlow
* Streamlit
* NumPy
* OpenCV
* MTCNN
* Pillow

## 👨‍💻 Pengembang

**Tegar Pratama**
Mahasiswa Teknik Informatika
Universitas Nusa Putra

## 📄 Lisensi
Proyek ini dibuat untuk keperluan riset dan pembelajaran, tidak untuk penggunaan komersial.
