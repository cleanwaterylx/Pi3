import pickle
import json
import os
import numpy as np
from tqdm import tqdm

n = 23781

with open(f'test_dataset_output/pairs_{n}.pkl', "rb") as f:
    all_pairs = pickle.load(f)
print(len(all_pairs))

data_root = '/home/disk8/dopp_data/visymscenes'
pair_with_intrinsics = []
for idx, pair in tqdm(enumerate(all_pairs)):
    image_0_relative_path, image_1_relative_path, pos_neg_pair_label, poses, full_image_list = pair
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

    pair = (image_0_relative_path, image_1_relative_path, pos_neg_pair_label, poses, full_image_list, [intrinsics1, intrinsics2])
    pair_with_intrinsics.append(pair)

with open(f"test_dataset_output/pairs_{n}_with_intrinsics.pkl", "wb") as f:
    pickle.dump(pair_with_intrinsics, f)

