import os
import json
import argparse
import numpy as np
import random
import datetime as dt
import copy

parser = argparse.ArgumentParser(description='Pembagian dataset')
parser.add_argument('--dataset_dir', required=True, help='Path ke anotasi dataset')
parser.add_argument('--test_percentage', type=int, default=10, help='Persentase data untuk testing')
parser.add_argument('--val_percentage', type=int, default=10, help='Persentase data untuk validation')
parser.add_argument('--nr_trials', type=int, default=10, help='Jumlah percobaan pembagian dataset')

args = parser.parse_args()
ann_input_path = os.path.join(args.dataset_dir, 'annotations.json')

# Memuat anotasi dataset
with open(ann_input_path, 'r') as f:
    dataset = json.load(f)

anns = dataset['annotations']
scene_anns = dataset['scene_annotations']
imgs = dataset['images']
jumlah_gambar = len(imgs)

jumlah_test = int(jumlah_gambar * args.test_percentage * 0.01 + 0.5)
jumlah_non_train = int(jumlah_gambar * (args.test_percentage + args.val_percentage) * 0.01 + 0.5)

for i in range(args.nr_trials):
    random.shuffle(imgs)

    # Struktur dataset baru
    train_set = {
        'info': dataset['info'],
        'images': [],
        'annotations': [],
        'scene_annotations': [],
        'licenses': dataset.get('licenses', []),
        'categories': dataset['categories'],
        'scene_categories': dataset['scene_categories'],
    }
    val_set = copy.deepcopy(train_set)
    test_set = copy.deepcopy(train_set)

    # Membagi gambar ke dalam set training, validation, dan testing
    test_set['images'] = imgs[:jumlah_test]
    val_set['images'] = imgs[jumlah_test:jumlah_non_train]
    train_set['images'] = imgs[jumlah_non_train:]

    # Membagi anotasi berdasarkan ID gambar
    def bagi_anotasi(source, target, img_ids):
        for ann in source:
            if ann['image_id'] in img_ids:
                target.append(ann)
    
    bagi_anotasi(anns, test_set['annotations'], [img['id'] for img in test_set['images']])
    bagi_anotasi(anns, val_set['annotations'], [img['id'] for img in val_set['images']])
    bagi_anotasi(anns, train_set['annotations'], [img['id'] for img in train_set['images']])
    
    bagi_anotasi(scene_anns, test_set['scene_annotations'], [img['id'] for img in test_set['images']])
    bagi_anotasi(scene_anns, val_set['scene_annotations'], [img['id'] for img in val_set['images']])
    bagi_anotasi(scene_anns, train_set['scene_annotations'], [img['id'] for img in train_set['images']])

    # Menyimpan dataset yang telah dibagi
    def simpan_json(data, filename):
        with open(os.path.join(args.dataset_dir, filename), 'w') as f:
            json.dump(data, f)
    
    simpan_json(train_set, f'annotations_{i}_train.json')
    simpan_json(val_set, f'annotations_{i}_val.json')
    simpan_json(test_set, f'annotations_{i}_test.json')