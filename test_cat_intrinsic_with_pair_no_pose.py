import pickle
import json
import os
import numpy as np
from tqdm import tqdm

n = 23781

dopp_pair = np.load(f'/home/disk8/dopp_data/pairs_metadata/test_pairs_visym.npy', allow_pickle=True)
# dopp_pair = np.concatenate([dopp_pair, dopp_pair[:, [1, 0, 2, 3]]], axis=0)

data_root = '/home/disk8/dopp_data/visymscenes'
pair_with_intrinsics = []
for idx, pair in tqdm(enumerate(dopp_pair)):
    image_0_relative_path, image_1_relative_path, pos_neg_pair_label, _ = pair
    scene = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
    base_path = os.path.join(data_root, scene)
    imgs = sorted([file for file in os.listdir(base_path) if file.endswith('.jpg')])

    image_0_name = image_0_relative_path.split('/')[-1]
    if image_0_name not in imgs or os.path.exists(os.path.join(data_root, image_1_relative_path)) is False:
        # print(f"Image {image_0_name} not found in {base_path}, skipping this pair.")
        continue

    metadata1 = json.load(open(os.path.join(data_root, image_0_relative_path.replace('.jpg', '.json')), 'r'))
    fx = metadata1['intrinsics']['fx']
    fy = metadata1['intrinsics']['fy']
    cx = metadata1['intrinsics']['cx']
    cy = metadata1['intrinsics']['cy']
    intrinsics1 = np.array([
    [fx,  0.0, cx],
    [0.0, fy,  cy],
    [0.0, 0.0, 1.0]], dtype=np.float32)

    metadata2 = json.load(open(os.path.join(data_root, image_1_relative_path.replace('.jpg', '.json')), 'r'))
    fx = metadata2['intrinsics']['fx']
    fy = metadata2['intrinsics']['fy']
    cx = metadata2['intrinsics']['cx']
    cy = metadata2['intrinsics']['cy']
    intrinsics2 = np.array([
    [fx,  0.0, cx],
    [0.0, fy,  cy],
    [0.0, 0.0, 1.0]], dtype=np.float32)

    pair = [image_0_relative_path, image_1_relative_path, pos_neg_pair_label, [intrinsics1, intrinsics2]]
    pair_with_intrinsics.append(pair)

pair_with_intrinsics = np.array(pair_with_intrinsics, dtype=object)
np.save(f"test_pairs_visym_with_intrinsics.npy", pair_with_intrinsics)

