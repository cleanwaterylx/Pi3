import torch
import argparse
from pi3.utils.basic import load_images_as_tensor_from_list
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import open3d as o3d
import numpy as np
import utils3d
import os
from collections import defaultdict
from tqdm import tqdm
import pickle

# batched calc camera poses from visymscenes dopp pairs with pi3 model

if __name__ == '__main__':
    dopp_pair_path = '/home/disk8/dopp_data/pairs_metadata/train_pairs_visym.npy'
    data_root='/home/disk8/dopp_data/visymscenes'
    dopp_pair = np.load(dopp_pair_path, allow_pickle=True)
    dopp_pair = np.concatenate([dopp_pair, dopp_pair[:, [1, 0, 2, 3]]], axis=0)  # symmetrized pair and double the data
    print(f'Number of DoppPairs: {len(dopp_pair)}')

    # 1. Prepare input data
    buckets = defaultdict(list)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for pair in tqdm(dopp_pair):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label, _ = pair
        scene = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
        base_path = os.path.join(data_root, scene)
        imgs = sorted([file for file in os.listdir(base_path) if file.endswith('.jpg')])

        image_0_name = image_0_relative_path.split('/')[-1]
        if image_0_name not in imgs or os.path.exists(os.path.join(data_root, image_1_relative_path)) is False:
            # print(f"Image {image_0_name} not found in {base_path}, skipping this pair.")
            continue
        idx = imgs.index(image_0_name)

        #  set image_0 as anchor ,idx +- 5
        step = 5
        idxs = list(range(max(0, idx - step), min(len(imgs), idx + step + 1)))

        selected_imgs = [os.path.join(base_path, imgs[j]) for j in idxs]
        if int(pos_neg_pair_label) == 1:
            selected_imgs.append(os.path.join(data_root, image_1_relative_path))
        N = len(selected_imgs)
        buckets[N].append((selected_imgs, (image_0_relative_path, image_1_relative_path, pos_neg_pair_label)))
    
    # 2. Prepare model
    print(f"Loading model...")
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = Pi3().to(device).eval()
    from safetensors.torch import load_file
    weight = load_file('ckpts/model.safetensors')
    model.load_state_dict(weight)
    
    # 3. batched process each bucket
    batch_size = 40
    all_pairs = []
    part = 1
    save_every = 5000
    for N in buckets:
        print(f"Processing bucket with {N} images, total {len(buckets[N])} pairs.")
        # process in batches
        for i in tqdm(range(0, len(buckets[N]), batch_size)):
            batch_image_lists = [item[0] for item in buckets[N][i:i+batch_size]]  # list of list of image paths
            batch_pair_info = [item[1] for item in buckets[N][i:i+batch_size]]  # list of (image_0_relative_path, image_1_relative_path)
            actual_batch_size = len(batch_image_lists)
            # flatten the list of lists
            flat_image_list = [img for sublist in batch_image_lists for img in sublist]
            # load images as tensor
            images_tensor = load_images_as_tensor_from_list(flat_image_list)
            images_tensor = images_tensor.to(device)  # [B*N, 3, H, W]
            B = actual_batch_size
            images_tensor = images_tensor.view(B, N, 3, images_tensor.shape[2], images_tensor.shape[3])  # [B, N, 3, H, W]

            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=dtype):
                    res = model(images_tensor)
            extrinsics = res['camera_poses'].cpu().numpy()  # [B, N, 4, 4] c2w

            # image_lists
            image_lists = [sublist for sublist in batch_image_lists]
            pairs = [(pair[0], pair[1], pair[2], extrinsics[i], image_lists[i]) for i, pair in enumerate(batch_pair_info)] 
            
            all_pairs.extend(pairs)
            if len(all_pairs) >= save_every:
                save_path = f"pairs_{part * save_every}.pkl"
                with open(save_path, "wb") as f:
                    pickle.dump(all_pairs, f)

                print(f"Saved {len(all_pairs)} pairs to {save_path}")
                part += 1
                all_pairs.clear() 

    if len(all_pairs) > 0:
        save_path = f"pairs_{part * save_every + len(all_pairs)}.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(all_pairs, f)

        print(f"Saved remaining {len(all_pairs)} pairs to {save_path}")

    
    
            





    
    


        




