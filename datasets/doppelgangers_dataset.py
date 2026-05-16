import os
import numpy as np
import os.path as osp
from PIL import Image
from datasets.base.transforms import *
import json
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset
import logging
import copy


class DoppelgangersDataset(BaseDataset):
    def __init__(
        self,
        data_root=None,
        verbose=False,
        max_distance=240,                    # 80
        **kwargs
    ):
        super().__init__(**kwargs)

        assert data_root is not None

        self.verbose = verbose
        self.dataset_label = 'Doppelgangers'
        mode = self.mode
        self.data_root = data_root
        self._rng = np.random.default_rng(42)

        self.meta_to_image = {
            'pairs_metadata/train_pairs_megadepth.npy': 'train_megadepth',
            'pairs_metadata/train_pairs_flip.npy': 'train_set_flip',
            'pairs_metadata/train_pairs_noflip.npy': 'train_set_noflip',
            'pairs_metadata/test_pairs.npy': 'test_set',
        }

        train_meta = [
            'pairs_metadata/train_pairs_megadepth.npy',
            'pairs_metadata/train_pairs_flip.npy',
            'pairs_metadata/train_pairs_noflip.npy'
        ]

        test_meta = ['pairs_metadata/test_pairs.npy']
        if self.mode == "train":
            self.dopp_pair = self._load_pairs(train_meta)
        else:
            self.dopp_pair = self._load_pairs(test_meta)
        
        status = "Training" if self.mode == "train" else "Testing"
        logging.info(f"{status}: Doppelgangers Data dataset length: {len(self)}")

    def _load_pairs(self, metadata_list):
        all_pairs = []

        for metadata in metadata_list:
            meta_path = metadata
            meta_path = os.path.join(self.data_root, meta_path)
            pairs = np.load(meta_path, allow_pickle=True)
    
            img_dir = self.meta_to_image[metadata]

            for pair in pairs:
                if 'gif' in pair[0] or 'gif' in pair[1]:
                    continue
                
                im1 = os.path.join(
                    'doppelgangers',
                    'images',
                    img_dir,
                    pair[0]
                )

                im2 = os.path.join(
                    'doppelgangers',
                    'images',
                    img_dir,
                    pair[1]
                )

                all_pairs.append([im1, im2, int(pair[2])])
        self._rng.shuffle(all_pairs)
        return all_pairs

    def __len__(self):
        return len(self.dopp_pair)

    def _get_views(self, index, resolution, rng):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label = self.dopp_pair[index]
        pos_neg_pair_label = int(pos_neg_pair_label)

        image_paths = [
            os.path.join(self.data_root, image_0_relative_path),
            os.path.join(self.data_root, image_1_relative_path),
        ]

        self.this_views_info = dict(
            scene=f'doppel_{index}',
            idxs=[0, 1],
        )

        label = int(rng.integers(0, 2))

        views = []
        for i, img_path in enumerate(image_paths):
            rgb_image = np.array(Image.open(img_path).convert('RGB'))
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]), dtype=np.float32)
            intrinsic = np.array([
                [1000, 0, rgb_image.shape[1] // 2],
                [0, 1000, rgb_image.shape[0] // 2],
                [0, 0, 1],
            ], dtype=np.float32)

            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsic, resolution, rng=rng, info=img_path)

            views.append(dict(
                img=rgb_image,
                pos_neg_pair_label=1 if pos_neg_pair_label else (label if i == 0 else 1 - label),
                dataset=self.dataset_label,
                label=os.path.basename(img_path),
                instance=str(i)
            ))

        if self.frame_num <= 2:
            return views[:self.frame_num]

        n1 = (self.frame_num + 1) // 2
        n2 = self.frame_num - n1

        expanded_views = []
        for _ in range(n1):
            expanded_views.append(copy.deepcopy(views[0]))
        for _ in range(n2):
            expanded_views.append(copy.deepcopy(views[1]))
        print(f"Expanded views for index {index}: {[(v['pos_neg_pair_label'], v['label']) for v in expanded_views]}")
        print(os.path.basename(image_0_relative_path), os.path.basename(image_1_relative_path), pos_neg_pair_label)
        return expanded_views
