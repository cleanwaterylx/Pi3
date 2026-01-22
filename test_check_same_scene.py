import os
import numpy as np

dopp_pair_path = '/home/disk8/dopp_data/pairs_metadata/train_pairs_visym.npy'
data_root='/home/disk8/dopp_data/visymscenes'
dopp_pair = np.load(dopp_pair_path, allow_pickle=True)
dopp_pair = dopp_pair[dopp_pair[:, 2] == '1']  # filter only negative pairs
same = diff = 0

for i, pair in enumerate(dopp_pair):
    image_0_relative_path, image_1_relative_path, pos_neg_pair_label, _ = pair
    image_0_scene = os.path.join(*image_0_relative_path.split('/')[:-1])  # get the scene from the first image path
    image_1_scene = os.path.join(*image_1_relative_path.split('/')[:-1])  # get the scene from the second image path
    # print(image_0_scene == image_1_scene, pos_neg_pair_label, image_0_relative_path, image_1_relative_path)
    if image_0_scene == image_1_scene:
        same += 1
    else:
        diff += 1

print(f"Same scene pairs: {same}, Different scene pairs: {diff}")
