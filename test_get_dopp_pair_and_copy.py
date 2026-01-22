import os
import numpy as np
import shutil

dopp_pair_path = '/home/disk8/dopp_data/pairs_metadata/train_pairs_visym.npy'
data_root='/home/disk8/dopp_data/visymscenes'
dopp_pair = np.load(dopp_pair_path, allow_pickle=True)
dopp_pair = dopp_pair[dopp_pair[:, 2] == '0']  # filter only negative pairs

rng = np.random.default_rng(42)  # 固定随机种子，方便复现；如不需要可去掉或改为 None
n_samples = 30


indices = rng.choice(len(dopp_pair), size=n_samples, replace=False)
sampled_pairs = dopp_pair[indices]

# print(sampled_pairs)
# input()

for i, pair in enumerate(sampled_pairs):
    image_0_relative_path, image_1_relative_path, pos_neg_pair_label, _ = pair
    scene = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
    base_path = os.path.join(data_root, scene)
    imgs = sorted([file for file in os.listdir(base_path) if file.endswith('.jpg')])
    # print(scene)
    # print(len(imgs))
    # print(imgs[0], imgs[1])

    image_0_name = image_0_relative_path.split('/')[-1]
    idx = imgs.index(image_0_name)
    print(pos_neg_pair_label)
    print(image_0_name)

    #  set image_0 as anchor ,idx random +- 5
    # todo random step or sample 
    step = 5
    idxs = list(range(max(0, idx - step), min(len(imgs), idx + step + 1)))

    imgs_to_copy = [os.path.join(base_path, imgs[i]) for i in idxs]
    # print('Images to copy:', imgs_to_copy)
    # input()
    save_dir = f'/home/disk3_SSD/ylx/dataset_pi3/visymscenes_test_{i}'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir+'/input', exist_ok=True)
    for img_path in imgs_to_copy:
        shutil.copy(img_path, save_dir+'/input/'+img_path.split('/')[-1])
    shutil.copy(os.path.join(data_root, image_1_relative_path), save_dir+'/input/'+image_1_relative_path.split('/')[-1])
    # input()


