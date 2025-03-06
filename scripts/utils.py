import numpy as np
import cv2
import pycocotools.mask as maskUtils

def annToMask(annotation, height, width):
    """
    Mengonversi anotasi COCO menjadi mask biner.
    annotation: Anotasi COCO dari objek dalam gambar.
    height, width: Ukuran gambar.
    
    Return:
    mask: Array [height, width] dengan nilai 1 untuk objek dan 0 untuk latar belakang.
    """
    rle = annToRLE(annotation, height, width)
    mask = maskUtils.decode(rle)
    return mask.astype(bool)

def annToRLE(annotation, height, width):
    """
    Mengubah anotasi dalam format poligon atau RLE tidak terkompresi menjadi RLE.
    annotation: Anotasi dari COCO.
    height, width: Ukuran gambar.

    Return:
    rle: Encoding dalam format RLE.
    """
    segm = annotation['segmentation']
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm['counts'], list):
        rle = maskUtils.frPyObjects(segm, height, width)
    else:
        rle = annotation['segmentation']
    return rle

def extract_bboxes(mask):
    """
    Menghitung bounding box dari mask objek.
    mask: Array [height, width] dengan nilai biner.

    Return:
    bbox: (y1, x1, y2, x2) dalam format [top-left, bottom-right].
    """
    pos = np.where(mask)
    if pos[0].size == 0:
        return np.array([0, 0, 0, 0])
    
    y1, x1 = np.min(pos, axis=1)
    y2, x2 = np.max(pos, axis=1)
    return np.array([y1, x1, y2, x2])

def resize_mask(mask, target_height, target_width):
    """
    Mengubah ukuran mask menggunakan interpolasi terdekat.
    mask: Mask asli [height, width].
    target_height, target_width: Ukuran mask baru.
    
    Return:
    mask yang telah diubah ukurannya.
    """
    return cv2.resize(mask.astype(np.uint8), (target_width, target_height), interpolation=cv2.INTER_NEAREST).astype(bool)

def replace_dataset_classes(dataset, class_map):
    """
    Mengganti label kelas dataset berdasarkan pemetaan class_map.
    dataset: Dataset COCO yang akan dimodifikasi.
    class_map: Dictionary yang memetakan kelas lama ke kelas baru.

    Return:
    Dataset yang telah diperbarui.
    """
    class_new_names = sorted(set(class_map.values()))
    class_originals = dataset['categories']
    dataset['categories'] = []
    class_ids_map = {}

    # Pastikan id 0 untuk 'Background'
    has_background = 'Background' in class_new_names
    if has_background:
        class_new_names.remove('Background')
        class_new_names.insert(0, 'Background')

    # Ganti kategori
    for id_new, class_new_name in enumerate(class_new_names):
        id_rectified = id_new if has_background else id_new + 1
        dataset['categories'].append({
            'supercategory': '',
            'id': id_rectified,
            'name': class_new_name,
        })
        for class_original in class_originals:
            if class_map[class_original['name']] == class_new_name:
                class_ids_map[class_original['id']] = id_rectified

    # Perbarui anotasi
    for ann in dataset['annotations']:
        ann['category_id'] = class_ids_map[ann['category_id']]

    return dataset
