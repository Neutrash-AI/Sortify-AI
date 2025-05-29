# SORTIFY-AI: Deteksi dan Klasifikasi Sampah

**SORTIFY-AI** adalah proyek deteksi dan klasifikasi jenis sampah berbasis kecerdasan buatan yang berjalan secara real-time menggunakan input dari kamera. Proyek ini mengintegrasikan berbagai arsitektur, seperti CNN Sequential, MobileNetV2, dan YOLO, untuk melakukan klasifikasi dan deteksi objek sampah.

## 🧠 Pendekatan Model AI

Proyek ini menggunakan kombinasi beberapa pendekatan AI untuk mendukung eksperimen dan optimasi performa:

- ✅ **CNN (Sequential)** untuk klasifikasi **2 kelas** (_organic_ vs _recycle_).
- ✅ **MobileNetV2** untuk klasifikasi **6 kelas** sampah.
- ✅ **YOLO (You Only Look Once)** untuk deteksi dan klasifikasi **real-time** objek dalam gambar, kompatibel dengan framework Ultralytics dan PyTorch.

> Dataset yang digunakan terdiri dari **±10.000 gambar**, mencakup berbagai jenis sampah untuk pelatihan dan evaluasi model.

## 🎯 Fitur Utama

- 🔍 Deteksi objek sampah secara **real-time** dari kamera laptop
- 📦 Visualisasi *bounding box* pada objek yang terdeteksi
- 🏷️ Menampilkan **label kelas** (misalnya: `BIODEGRADABLE`, `CARDBOARD`, dll.) dan **confidence score**
- 🧾 Mencetak **log deteksi** ke konsol jika confidence score melebihi ambang batas
- ⚙️ **Threshold deteksi** dapat diatur sesuai kebutuhan untuk menyaring hasil dengan confidence rendah

## 🗂️ Struktur Direktori Proyek

```
SORTIFY-AI/
├── Deteksi/
│   ├── Percobaan 1/
│   │   ├── best (2).onnx
│   │   ├── best (5).pt
│   │   ├── best_float16 (3).tflite
│   │   ├── best_float32 (2).tflite
│   │   └── Sortify_YOLO.ipynb
│   └── Percobaan 2/
│       ├── best (3).onnx
│       ├── best (6).pt              # <-- Model yang digunakan dalam contoh ini
│       ├── best_float16 (4).tflite
│       ├── best_float32 (3).tflite
│       └── Sortify_YOLO_Lanjutan.ipynb
├── Klasifikasi/
│   ├── Dataset dua kelas/
│   │   ├── sortify_ai.h5
│   │   └── Sortify_Klasifikasi_v1.ipynb
│   └── Dataset enam kelas/
│       ├── sortify_ai_v2.h5
│       └── Sortify_Klasifikasi_v2.ipynb
├── src/
│   └── main.py
├── venv/
├── .gitignore
├── README.md
└── requirements.txt
```

**Catatan:** Path model pada skrip `main.py` saat ini diatur ke `"./Deteksi/Percobaan 2/best (6).pt"`. Jika ingin memindahkan model ke direktori lain (misalnya, ke folder `model/` di root proyek), pastikan untuk memperbarui variabel `MODEL_PATH` di dalam `src/main.py`. Skrip `main.py` hanya digunakan untuk deteksi objek dengan YOLO.

## 🚀 Setup dan Instalasi

### 1. Persiapan Awal

Clone repositori ini ke perangkat pengguna:

```bash
git clone https://github.com/Neutrash-AI/Sortify-AI
cd SORTIFY-AI
```

### 2. Membuat dan Mengaktifkan Virtual Environment

Menggunakan virtual environment sangat disarankan untuk menjaga dependensi proyek tetap terisolasi.

#### **Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

#### **macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> 💡 **Tips:** Setelah berhasil diaktifkan, maka akan melihat `(venv)` di awal prompt terminal.

### 3. Instalasi Dependensi

Pastikan virtual environment sudah aktif, kemudian install semua dependensi yang dibutuhkan:

```bash
pip install -r requirements.txt
```

### 4. Menjalankan Aplikasi

1. **Verifikasi Path Model**
   
   Pastikan file model YOLO (`best (6).pt`) berada di lokasi yang sesuai dengan variabel `MODEL_PATH` di file `src/main.py`:
   
   ```python
   MODEL_PATH = "./Deteksi/Percobaan 2/best (6).pt"
   ```
   
   Jika menggunakan model dengan nama atau lokasi berbeda, ubah nilai `MODEL_PATH` sesuai kebutuhan.

2. **Jalankan Aplikasi**
   
   Eksekusi skrip utama dari direktori root proyek:
   
   ```bash
   python src/main.py
   ```

3. **Menggunakan Aplikasi**
   
   - Aplikasi akan mencoba menginisialisasi kamera laptop default
   - Jendela video stream akan muncul jika berhasil
   - Objek sampah yang terdeteksi akan ditandai dengan:
     - **Bounding box berwarna hijau**
     - **Label kelas** (contoh: `BIODEGRADABLE`, `CARDBOARD`)
     - **Confidence score** (jika di atas threshold)
   - Tekan tombol **`q`** untuk keluar dari aplikasi

## 🛠️ Troubleshooting

### ❌ **Error: `NameError: name 'YOLO' is not defined`**
```bash
pip install ultralytics
```

### ❌ **Error: `ModuleNotFoundError: No module named 'torch'`**
```bash
pip install torch torchvision torchaudio
```

### ❌ **Kamera tidak terbuka**
- Pastikan kamera tidak sedang digunakan aplikasi lain
- Coba ubah parameter kamera di `cv2.VideoCapture(0)` menjadi `1`, `2`, dst.
- Periksa driver kamera di sistem operasi pengguna.

### ❌ **Model tidak ditemukan**
- Verifikasi path di variabel `MODEL_PATH` dalam `src/main.py`
- Pastikan file model tidak korup
- Gunakan path absolut jika perlu

### ❌ **Hasil prediksi tidak akurat**
- **Pencahayaan:** Pastikan kondisi pencahayaan cukup baik
- **Fokus:** Posisikan objek dengan jarak dan fokus yang optimal
- **Threshold:** Sesuaikan nilai `CONFIDENCE_DISPLAY_THRESHOLD` di `main.py`

**Dibuat dengan ❤️ oleh Tim Neutrash-AI**