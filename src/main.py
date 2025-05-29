import os
import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Model AI
MODEL_PATH = "./Deteksi/Percobaan 2/best (6).pt"  # Path ke model PyTorch
USE_MODEL = os.path.exists(MODEL_PATH)
model_yolo = None

if USE_MODEL:
    print(f"Model YOLO (.pt) ditemukan -> {MODEL_PATH}")
    try:
        model_yolo = YOLO(MODEL_PATH)
        print("Model PyTorch YOLO berhasil dimuat.")
    except Exception as e:
        print(f"Gagal memuat model AI: {e}")
        USE_MODEL = False
        model_yolo = None
else:
    print("Tidak ada model AI.")

# Fungsi inferensi menggunakan Ultralytics YOLO
def run_inference_yolo(model, frame_input, confidence_threshold=0.5): # Tambahkan confidence_threshold
    detections_result = []
    results = model(frame_input)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            conf = float(box.conf[0])
            # HANYA TAMBAHKAN DETEKSI JIKA CONFIDENCE DI ATAS THRESHOLD
            if conf >= confidence_threshold:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                if hasattr(model, 'names') and 0 <= class_id < len(model.names):
                    label = model.names[class_id]
                else:
                    label = f"ClassID_{class_id}"

                detections_result.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2 - x1, y2 - y1]
                })
    return detections_result

# Gambar box pada frame
def draw_bounding_box(frame, obj):
    x, y, w, h = obj['bbox']
    color = (0, 255, 0)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{obj['label']} ({obj['confidence'] * 100:.1f}%)"
    cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


# Proses utama menggunakan kamera laptop
def main_loop_laptop_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Tidak dapat membuka kamera laptop. Pastikan kamera tidak digunakan oleh aplikasi lain.")
        return

    print("Kamera laptop berhasil diinisialisasi. Tekan 'q' untuk keluar.")

    last_print_time = time.time()
    print_interval = 5 
    CONFIDENCE_DISPLAY_THRESHOLD = 0.5 # Ambang batas confidence

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera laptop.")
            break

        detections_to_display_and_log = [] # Untuk menyimpan deteksi yang memenuhi threshold
        frame_to_display = frame.copy()
        current_time = time.time()

        if USE_MODEL and model_yolo is not None:
            try:
                all_detections = run_inference_yolo(model_yolo, frame, confidence_threshold=0.0) # Dapatkan semua deteksi awal

                for obj in all_detections:
                    # Filter berdasarkan confidence SEBELUM menggambar atau mencatat
                    if obj['confidence'] >= CONFIDENCE_DISPLAY_THRESHOLD:
                        detections_to_display_and_log.append(obj) # Simpan untuk digambar dan dilog
                        draw_bounding_box(frame_to_display, obj)

                if detections_to_display_and_log and (current_time - last_print_time >= print_interval):
                    print(f"--- Hasil Deteksi YOLO (Confidence >= {CONFIDENCE_DISPLAY_THRESHOLD*100}%) ({time.strftime('%H:%M:%S')}) ---")
                    for i, det_obj in enumerate(detections_to_display_and_log):
                        print(f"Objek {i+1}: Label = {det_obj['label']}, Confidence = {det_obj['confidence']*100:.1f}%, Bbox = {det_obj['bbox']}")
                    last_print_time = current_time
                    print("-------------------------------------------------------------------")

            except Exception as e:
                print(f"Error saat inferensi YOLO atau menggambar bounding box: {e}")
        else:
            if (current_time - last_print_time >= print_interval):
                print(f"Tidak ada model AI aktif atau model gagal dimuat. Menampilkan frame mentah. ({time.strftime('%H:%M:%S')})")
                last_print_time = current_time

        cv2.imshow("Laptop Camera Stream - YOLO Detection", frame_to_display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Streaming kamera laptop dihentikan.")

if __name__ == "__main__":
    main_loop_laptop_camera()