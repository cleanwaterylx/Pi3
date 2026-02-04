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
        scene1 = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
        scene2 = os.path.join(*image_1_relative_path.split('/')[:3])
        base_path = os.path.join(self.data_root, scene1)
        imgs = sorted([file for file in os.listdir(base_path) if file.endswith('.jpg')])
        # print(scene)
        # print(len(imgs))
        # print(imgs[0], imgs[1])

        image_0_name = image_0_relative_path.split('/')[-1]
        image_1_name = image_1_relative_path.split('/')[-1]
        idx = imgs.index(image_0_name)
        # print(image_0_name)
        # print('Image 0 index in the sequence:', idx)

        #  set image_0 as anchor ,idx random +- 5
        # todo random step or sample 
        step = (self.frame_num - 2) // 2
        idxs = list(range(max(0, idx - step), min(len(imgs), idx + step + 1)))
        # print(self.frame_num)
        # print(len(idxs))

        self.this_views_info = dict(
            scene=scene1,
            idxs=idxs,
        )


        views = []
        for i, idx in enumerate(idxs):
            img_path = os.path.join(base_path, imgs[idx])
            
            rgb_image = np.array(Image.open(img_path))
            # todo crop image not PIL.Image.Image 
            # depthmap fill dummy
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics[0].copy(), resolution, rng=rng, info=img_path)

            views.append(dict(
                img=rgb_image,
                pos_neg_pair_label = 1,
                dataset=self.dataset_label,
                label=imgs[idx],
                instance=str(idx)
            ))
        while len(views) < self.frame_num:
            views.append(copy.deepcopy(views[-1]))   #todo  choose one view pad to ensure enough views 
        
        if int(pos_neg_pair_label) == 1:
            rgb_image = np.array(Image.open(os.path.join(self.data_root, image_1_relative_path)))
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics[1].copy(), resolution, rng=rng, info=os.path.join(self.data_root, image_1_relative_path))
            views.append(dict(
                img=rgb_image,
                pos_neg_pair_label = 1,
                dataset=self.dataset_label,
                label=os.path.join(self.data_root, image_1_relative_path),
                instance=str(idx)
            ))

        if int(pos_neg_pair_label) == 0:
            rgb_image = np.array(Image.open(os.path.join(self.data_root, image_1_relative_path)))
            depthmap = np.ones((rgb_image.shape[0], rgb_image.shape[1]),dtype=np.float32)
            rgb_image, depthmap, intrinsic_ = self._crop_resize_if_necessary(
                rgb_image, depthmap, intrinsics[1].copy(), resolution, rng=rng, info=os.path.join(self.data_root, image_1_relative_path))
            views.append(dict(
                img=rgb_image,
                pos_neg_pair_label = 0,
                dataset=self.dataset_label,
                label=os.path.join(self.data_root, image_1_relative_path),
                instance=str(idx)
            ))
        
        return views   

