import os
import json
import numpy as np
import copy
import utils

from PIL import Image, ExifTags
from pycocotools.coco import COCO

class Taco(utils.Dataset):
    """
    Kelas untuk memuat dataset TACO dan mengubahnya ke format yang dapat digunakan oleh Mask R-CNN.
    """

    def load_taco(self, dataset_dir, round, subset, class_ids=None,
                  class_map=None, return_taco=False):
        """
        Memuat subset dari dataset TACO.
        dataset_dir: Direktori utama dataset.
        round: Nomor pembagian dataset.
        subset: Subset yang dimuat (train, val, test).
        class_ids: Jika diberikan, hanya memuat gambar dengan kelas tertentu.
        class_map: Pemetaan kelas asli ke sistem kelas baru.
        return_taco: Jika True, mengembalikan objek COCO.
        """
        ann_filepath = os.path.join(dataset_dir, f'annotations_{round}_{subset}.json')
        assert os.path.isfile(ann_filepath)

        # Memuat dataset
        dataset = json.load(open(ann_filepath, 'r'))
        self.replace_dataset_classes(dataset, class_map)
        taco_coco = COCO()
        taco_coco.dataset = dataset
        taco_coco.createIndex()

        # Menambahkan kelas
        image_ids = []
        background_id = -1
        class_ids = sorted(taco_coco.getCatIds())
        for i in class_ids:
            class_name = taco_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                self.add_class("taco", i, class_name)
                image_ids.extend(list(taco_coco.getImgIds(catIds=i)))
            else:
                background_id = i
        image_ids = list(set(image_ids))
        if background_id > -1:
            class_ids.remove(background_id)

        print('Jumlah gambar yang digunakan:', len(image_ids))

        # Menambahkan gambar
        for i in image_ids:
            self.add_image(
                "taco", image_id=i,
                path=os.path.join(dataset_dir, taco_coco.imgs[i]['file_name']),
                width=taco_coco.imgs[i]["width"],
                height=taco_coco.imgs[i]["height"],
                annotations=taco_coco.loadAnns(taco_coco.getAnnIds(imgIds=[i], catIds=class_ids)))
        
        if return_taco:
            return taco_coco

    def load_image(self, image_id):
        """
        Memuat gambar berdasarkan ID dan mengembalikannya sebagai array numpy.
        """
        image = Image.open(self.image_info[image_id]['path'])
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)
        return np.array(image)

    def load_mask(self, image_id):
        """
        Memuat instance mask untuk gambar yang diberikan.
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]
        
        for annotation in annotations:
            class_id = self.map_source_class_id(f"taco.{annotation['category_id']}")
            if class_id:
                mask = utils.annToMask(annotation, image_info["height"], image_info["width"])
                if mask.max() < 1:
                    continue
                if annotation['iscrowd']:
                    class_id *= -1
                instance_masks.append(mask)
                class_ids.append(class_id)
        
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            return super(Taco, self).load_mask(image_id)
    
    def replace_dataset_classes(self, dataset, class_map):
        """
        Mengubah kelas dataset berdasarkan dictionary class_map.
        """
        class_new_names = sorted(set(class_map.values()))
        class_originals = copy.deepcopy(dataset['categories'])
        dataset['categories'] = []
        class_ids_map = {}

        has_background = 'Background' in class_new_names
        if has_background:
            class_new_names.remove('Background')
            class_new_names.insert(0, 'Background')

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

        for ann in dataset['annotations']:
            ann['category_id'] = class_ids_map[ann['category_id']]
