import os
import numpy as np
import os.path as osp
from PIL import Image
from datasets.base.transforms import *
import json
from tqdm import tqdm
from datasets.base.base_dataset import BaseDataset
import pickle
import copy
import random

class VisymScenesDataset(BaseDataset):
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
        self.dataset_label = 'VisymScenes'
        mode = self.mode
        self.data_root = data_root

        self.train_dopp_pair_path = 'pair_data/train_pairs_visym_with_intrinsics.npy'     #todo self.dopp_pair 是复杂嵌套（字符串、numpy 矩阵、列表、字典等）， 转化成dic
        self.test_dopp_pair_path = 'pair_data/test_pairs_visym_with_intrinsics.npy'
        # todo only symmetrized negative pair
        # pair example 
        # ['siteSTR0003/e3ed7db9-1988-4178-bb45-6ada3ecc39c8-iPhone_7/2023-09-12_20-28-08/siteSTR0003-e3ed7db9-1988-4178-bb45-6ada3ecc39c8-iPhone_7-2023-09-12_20-28-08-000012.jpg'
        # 'siteSTR0003/e3ed7db9-1988-4178-bb45-6ada3ecc39c8-iPhone_7/2023-09-12_20-28-08/siteSTR0003-e3ed7db9-1988-4178-bb45-6ada3ecc39c8-iPhone_7-2023-09-12_20-28-08-000015.jpg'
        # '1', pose : (N, 4, 4) np.array]

        if mode == 'train':
            self.dopp_pair = np.load(self.train_dopp_pair_path, allow_pickle=True)
        else:
            self.dopp_pair = np.load(self.test_dopp_pair_path, allow_pickle=True)

        if self.verbose:
            print(f'[{self.dataset_label}] Number of DoppPairs: {len(self.dopp_pair)}')
        print(f'[{self.dataset_label}] Number of DoppPairs: {len(self.dopp_pair)}')


        


        # self.dopp_pair = self.dopp_pair


    def __len__(self):
        return len(self.dopp_pair)
    
    def _get_views(self, index, resolution, rng):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics = self.dopp_pair[index]
        pos_neg_pair_label = int(pos_neg_pair_label)
        scene1 = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
        scene2 = os.path.join(*image_1_relative_path.split('/')[:3])
        base_path1 = os.path.join(self.data_root, scene1)
        base_path2 = os.path.join(self.data_root, scene2)
        imgs_1 = sorted([file for file in os.listdir(base_path1) if file.endswith('.jpg')])
        imgs_2 = sorted([file for file in os.listdir(base_path2) if file.endswith('.jpg')])
        # print(scene)
        # print(len(imgs))
        # print(imgs[0], imgs[1])

        image_0_name = image_0_relative_path.split('/')[-1]
        image_1_name = image_1_relative_path.split('/')[-1]
        idx_1 = imgs_1.index(image_0_name)
        idx_2 = imgs_2.index(image_1_name)
        # print(image_0_name)
        # print('Image 0 index in the sequence:', idx)

        def sample_indices(center_idx, total_len, num_samples):
            if num_samples == 1:
                return [center_idx]

            indices = [center_idx]
            offset = 1

            while len(indices) < num_samples:
                left = center_idx - offset
                right = center_idx + offset

                if left >= 0:
                    indices.append(left)
                    if len(indices) >= num_samples:
                        break

                if right < total_len:
                    indices.append(right)
                    if len(indices) >= num_samples:
                        break

                if left < 0 and right >= total_len:
                    break

                offset += 1

            return sorted(indices)

        split_choices = [(n1, self.frame_num - n1) for n1 in range(self.frame_num // 4, self.frame_num - self.frame_num // 4 + 1)]
        n1, n2 = random.choice(split_choices)

        idxs_1 = sample_indices(idx_1, len(imgs_1), n1)
        idxs_2 = sample_indices(idx_2, len(imgs_2), n2)
        # print(self.frame_num)
        # print(max(0, idx_1 - step), min(len(imgs_1), idx_1 + step))
        # print('idx_1:', idx_1, 'idx_2:', idx_2)
        # print('idxs_1:', idxs_1, 'idxs_2:', idxs_2)
        # print(len(idxs))

        self.this_views_info = dict(
            scene=scene1,
            idxs=idxs_1 + idxs_2,
        )

        label = random.randint(0, 1)
        
        views = []
        for i, idx_1 in enumerate(idxs_1):
            img_path = os.path.join(base_path1, imgs_1[idx_1])
            
            rgb_image = np.array(Image.open(img_path))
            # todo crop image not PIL.Image.Image 
            # depthmap fill dummy
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics[0].copy(), resolution, rng=rng, info=img_path)

            views.append(dict(
                img=rgb_image,
                pos_neg_pair_label = 1 if pos_neg_pair_label else label,
                dataset=self.dataset_label,
                label=imgs_1[idx_1],
                instance=str(idx_1)
            ))
        
        for i, idx_2 in enumerate(idxs_2):
            img_path = os.path.join(base_path2, imgs_2[idx_2])
            
            rgb_image = np.array(Image.open(img_path))
            # todo crop image not PIL.Image.Image 
            # depthmap fill dummy
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics[1].copy(), resolution, rng=rng, info=img_path)

            views.append(dict(
                img=rgb_image,
                pos_neg_pair_label = 1 if pos_neg_pair_label else 1 - label,   # different from the first image
                dataset=self.dataset_label,
                label=imgs_2[idx_2],
                instance=str(idx_2)
            ))
        while len(views) < self.frame_num:
            views.append(copy.deepcopy(views[-1]))   #todo  choose one view pad to ensure enough views 
        
        # if int(pos_neg_pair_label) == 1:
        #     rgb_image = np.array(Image.open(os.path.join(self.data_root, image_1_relative_path)))
        #     depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
        #     rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
        #         rgb_image, depthmap, intrinsics[1].copy(), resolution, rng=rng, info=os.path.join(self.data_root, image_1_relative_path))
        #     views.append(dict(
        #         img=rgb_image,
        #         pos_neg_pair_label = 1,
        #         dataset=self.dataset_label,
        #         label=os.path.join(self.data_root, image_1_relative_path),
        #         instance=str(idx_1)
        #     ))

        # if int(pos_neg_pair_label) == 0:
        #     rgb_image = np.array(Image.open(os.path.join(self.data_root, image_1_relative_path)))
        #     depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
        #     rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
        #         rgb_image, depthmap, intrinsics[1].copy(), resolution, rng=rng, info=os.path.join(self.data_root, image_1_relative_path))
        #     views.append(dict(
        #         img=rgb_image,
        #         pos_neg_pair_label = 0,
        #         dataset=self.dataset_label,
        #         label=os.path.join(self.data_root, image_1_relative_path),
        #         instance=str(idx_1)
        #     ))
        
        return views   
