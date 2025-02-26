# 📦 Sortify-AI

🚀 **Sortify-AI** adalah modul yang digunakan untuk mengunduh, mengelola, dan memproses dataset gambar untuk deteksi sampah otomatis. Proyek ini memanfaatkan anotasi dataset untuk model klasifikasi dan segmentasi sampah.

---

## 📥 Persyaratan Instalasi

Pastikan Anda telah menginstal **Python** sebelum melanjutkan. Kemudian, jalankan perintah berikut untuk menginstal semua dependensi yang diperlukan:

```sh
pip install -r requirements.txt
```

---

## 🔹 Cara Mengunduh Dataset

Jalankan perintah berikut untuk mengunduh dataset gambar:

```sh
python download.py
```

Atau, Anda dapat mengunduhnya secara manual melalui Zenodo dengan mengklik tautan berikut:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3587843.svg)](https://doi.org/10.5281/zenodo.3587843)

---

## 📌 Struktur Proyek

```
📦 sortify-ai
├── 📂 data/            # Folder penyimpanan dataset dan anotasi
│   ├── annotations.json  # File anotasi untuk dataset
├── 📂 notebooks/       # Notebook Jupyter untuk eksplorasi dan analisis data
├── 📂 scripts/         # Skrip Python untuk pengolahan data
├── 📜 download.py      # Skrip untuk mengunduh dataset dari sumber
├── 📜 requirements.txt # Daftar dependensi Python
├── 📜 README.md        # Dokumentasi proyek
└── 📜 .gitignore       # File & folder yang dikecualikan dari Git
```

<br>

---

❓ **Bagaimana jika proses unduhan dataset gagal?**  
✅ Coba jalankan kembali skrip `download.py`. Jika masih gagal, unduh dataset secara manual melalui Zenodo dan simpan di folder `data/`.

---