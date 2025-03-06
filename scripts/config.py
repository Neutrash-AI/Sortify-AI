import math
import numpy as np

class Konfigurasi(object):
    """
    Kelas dasar untuk konfigurasi model Mask R-CNN.
    Untuk menyesuaikan model, buat sub-kelas dari kelas ini dan ubah parameter yang diperlukan.
    """
    
    # Nama konfigurasi (ubah sesuai kebutuhan)
    NAME = None  
    
    # Penggunaan GPU (1 untuk CPU)
    GPU_COUNT = 1
    
    # Jumlah gambar yang diproses per GPU dalam satu iterasi
    IMAGES_PER_GPU = 2
    
    # Jumlah langkah dalam satu epoch
    STEPS_PER_EPOCH = 1000
    
    # Jumlah langkah validasi setiap akhir epoch
    VALIDATION_STEPS = 50
    
    # Arsitektur backbone yang digunakan (resnet50 atau resnet101)
    BACKBONE = "resnet50"
    
    # Optimizer yang digunakan
    OPTIMIZER = 'SGD'
    
    # Ukuran skala fitur backbone
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]
    
    # Jumlah kelas (termasuk latar belakang)
    NUM_CLASSES = 1  
    
    # Ukuran anchor untuk deteksi objek
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    
    # Rasio aspek untuk setiap anchor (lebar/tinggi)
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]
    
    # Ambang batas NMS untuk penyaringan proposal
    RPN_NMS_THRESHOLD = 0.7
    
    # Ukuran minimum dan maksimum gambar
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    
    # Mode perubahan ukuran gambar
    IMAGE_RESIZE_MODE = "square"
    
    # Tingkat pembelajaran
    LEARNING_RATE = 0.001
    
    # Momentum pembelajaran
    LEARNING_MOMENTUM = 0.9
    
    # Regularisasi bobot
    WEIGHT_DECAY = 0.0001
    
    # Ukuran batch efektif
    def __init__(self):
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        
        if self.IMAGE_RESIZE_MODE == "crop":
            self.IMAGE_SHAPE = np.array([self.IMAGE_MIN_DIM, self.IMAGE_MIN_DIM, 3])
        else:
            self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])
        
        # Ukuran metadata gambar
        self.IMAGE_META_SIZE = 1 + 3 + 3 + 4 + 1 + self.NUM_CLASSES
    
    # Fungsi untuk menampilkan konfigurasi
    def display(self):
        print("\nKonfigurasi:")
        for atribut in dir(self):
            if not atribut.startswith("__") and not callable(getattr(self, atribut)):
                print("{:30} {}".format(atribut, getattr(self, atribut)))
        print("\n")
