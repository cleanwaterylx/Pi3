import torch
import torch.nn.functional as F
import argparse
from pi3.utils.basic import load_images_as_tensor_from_list
from pi3.utils.geometry import depth_edge
from pi3.models.pi3_classification_one_image_one_feature_with_infonce_loss import Pi3
import open3d as o3d
import numpy as np
import utils3d
import os
from collections import defaultdict
from tqdm import tqdm
import pickle
import random
from sklearn.metrics import average_precision_score,roc_auc_score, roc_curve, precision_recall_curve
import PIL.Image as Image
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image

# batched test camera poses from visymscenes dopp pairs with pi3 model

def class_to_binary(pred_class, N):
    """
    pred_class: LongTensor, shape (B,)
                value in [0, N]  (N 表示 all good)
    return:     LongTensor, shape (B, N)
    """
    B = pred_class.shape[0]
    out = torch.ones(B, N, device=pred_class.device, dtype=torch.long)

    mask = pred_class < N          # (B,)
    out[mask, pred_class[mask]] = 0

    return out


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    data_root='/home/disk8/dopp_data/visymscenes'
    dopp_pair = np.load('pair_data/test_pairs_visym_with_intrinsics.npy', allow_pickle=True)
    # shuffle the pairs
    np.random.seed(42) 
    np.random.shuffle(dopp_pair)
    print(f'Number of test DoppPairs: {len(dopp_pair)}')
    
    # all_pair_imgs = []  # 用来收集每个 pair 的图片
    # all_labels = []  # 用来收集每个 pair 的标签

    # count = 0
    # id = 0
    # for pair in tqdm(dopp_pair):
    #     image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics = pair
    #     scene = os.path.join(*image_0_relative_path.split('/')[:3])
    #     base_path = os.path.join(data_root, scene)
    #     imgs = sorted([file for file in os.listdir(base_path) if file.endswith('.jpg')])

    #     image_0_name = image_0_relative_path.split('/')[-1]
    #     if image_0_name not in imgs or not os.path.exists(os.path.join(data_root, image_1_relative_path)):
    #         continue

    #     idx = imgs.index(image_0_name)
    #     step = 2
    #     idxs = list(range(max(0, idx - step), min(len(imgs), idx + step + 1)))
    #     selected_imgs = [os.path.join(base_path, imgs[j]) for j in idxs]
    #     selected_imgs.append(os.path.join(data_root, image_1_relative_path))

    #     # 将这个 pair 的图片加入 all_pair_imgs
    #     pair_tensors = [torch.from_numpy(np.array(Image.open(p).convert('RGB'))).permute(2,0,1) 
    #                     for p in selected_imgs]
    #     # 获取图片尺寸
    #     if len(pair_tensors) > 0:
    #         C, H, W = pair_tensors[0].shape
    #     else:
    #         # 防止 selected_imgs 空的情况，默认 3x256x256
    #         C, H, W = 3, 256, 256

    #     # 不足 target_num 补全全黑图
    #     while len(pair_tensors) < 6:
    #         black_img = torch.zeros(C, H, W, dtype=torch.uint8)
    #         pair_tensors.append(black_img)
    #     all_pair_imgs.append(pair_tensors)
    #     all_labels.append(pos_neg_pair_label)

    #     count += 1
    #     if count == 5:  # 收集 5 个 pair 后显示
    #         # 将每个 pair 的图片平铺
    #         print(all_labels)  # 打印标签
    #         grid_imgs = []
    #         for pair_imgs in all_pair_imgs:
    #             grid_imgs.extend(pair_imgs)
    #         grid = make_grid(grid_imgs, nrow=len(all_pair_imgs[0]))  # 每行显示一个 pair 的所有图片
    #         # plt.figure(figsize=(20, 8))
    #         # plt.imshow(grid.permute(1,2,0))
    #         # plt.axis('off')
    #         # plt.show()
            
    #          # 保存到文件
    #         save_path = f'imgs/pair_grid_{id}_labal_{all_labels}.jpg'  # batch_idx 可以自己维护计数
    #         grid_float = grid.float() / 255.0
    #         save_image(grid_float, save_path)
    #         print(f"Saved grid to {save_path}")

    #         # 清空，准备下一批 5 个 pair
    #         all_pair_imgs = []
    #         all_labels = []
    #         count = 0
    #         id += 1
            
    # quit()


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    
    # 2. Prepare model
    print(f"Loading model...")
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = Pi3().to(device).eval()
    from safetensors.torch import load_file
    weight = load_file('ckpts/pi3_visymscenes_feature_512_epoch3.safetensors')
    pi3_weight = load_file('ckpts/model.safetensors')
    #load conf weights from pi3_weight
    conf_decoder_weight = {
        k.replace('model.conf_decoder.', ''): pi3_weight[k] for k in pi3_weight.keys() if k.startswith('conf_decoder.')
    }
    conf_head_weight = {
        k.replace('model.conf_head.', ''): pi3_weight[k] for k in pi3_weight.keys() if k.startswith('conf_head.')
    }
    weight.update(conf_decoder_weight)
    weight.update(conf_head_weight)
    
    model.load_state_dict(weight)
    
    # 3. batched process each bucket
    batch_size = 10
    all_pairs = []
    part = 1
    save_every = 5000
    
    gts = []
    preds = []
    false_pair = []
    for pair in tqdm(dopp_pair):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics = pair
        pos_neg_pair_label = int(pos_neg_pair_label)
        scene1 = os.path.join(*image_0_relative_path.split('/')[:3])  # get the scene from the first image path
        scene2 = os.path.join(*image_1_relative_path.split('/')[:3])
        base_path1 = os.path.join(data_root, scene1)
        base_path2 = os.path.join(data_root, scene2)
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

        #  set image_0 as anchor ,idx random +- 5
        # todo random step or sample 
        step = 4
        idxs_1 = list(range(max(0, idx_1 - step), min(len(imgs_1), idx_1 + step + 1)))
        idxs_2 = list(range(max(0, idx_2 - step), min(len(imgs_2), idx_2 + step + 1)))

        selected_imgs = [os.path.join(base_path1, imgs_1[j]) for j in idxs_1]
        labels = [1] * len(selected_imgs)
        selected_imgs += [os.path.join(base_path2, imgs_2[j]) for j in idxs_2]
        labels += [1 if pos_neg_pair_label else 0] * len(idxs_2)
        
        # combined = list(zip(selected_imgs, labels))

        # np.random.default_rng(42).shuffle(combined)
        
        # selected_imgs, labels = map(list, zip(*combined))
         
        # load images as tensor
        images_tensor = load_images_as_tensor_from_list(selected_imgs)
        images_tensor = images_tensor.to(device)
        

        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=dtype):
                res = model(images_tensor[None]) # Add batch dimension
        # print(res['logits'].shape)
        features = res['feat']   # [B, N, C]
        # 1) normalize features
        features = F.normalize(features, dim=-1)

        # 2) pairwise similarity: [B, N, N]
        sim = torch.matmul(features, features.transpose(1, 2))

        # print(selected_imgs)
        # print(features)
        # print(features.shape)
        # print(sim.shape)
        print(sim)
        print(labels)
        input()

        gts.append(pos_neg_pair_label)
        preds.append(0 if sim[0][-1][0] < 12 else 1)  # 12 is a threshold, can be tuned based on validation set

        # print(pos_neg_pair_label, 0 if sim[0][-1][0] < 12 else 1)
        # input()

        # for i, pair in enumerate(batch_pair_info):
        #     image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics = pair
        #     gt = int(pos_neg_pair_label)
        #     gts.append(gt)
        #     preds.append(pred[i])
        #     if gt == 1 and pred[i] < 0.5:
        #         false_pair.append((image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics))
        #     if gt == 0 and pred[i] >= 0.5:
        #         false_pair.append((image_0_relative_path, image_1_relative_path, pos_neg_pair_label, intrinsics))
                    
    np.save('gts_preds_visym_test_pi3_visymscenes_feature_512_epoch3.npy', {'gts': gts, 'preds': preds})        
    
    ap = average_precision_score(gts, preds)
    auc = roc_auc_score(gts, preds)
    print(f"AP: {ap:.4f}, AUC: {auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(gts, preds)

    # Prec@Recall>=0.85
    target_recall = 0.85
    idx = np.where(recall >= target_recall)[0]
    prec_at_recall = np.max(precision[idx]) if len(idx) > 0 else 0.0
    print("Prec@Recall>=0.85:", prec_at_recall)

    # Recall@Prec>=0.99
    target_precision = 0.99
    idx = np.where(precision >= target_precision)[0]
    recall_at_prec = np.max(recall[idx]) if len(idx) > 0 else 0.0
    print("Recall@Prec>=0.99:", recall_at_prec)
    
    np.save('gts_preds_visym_test_pi3_visymscenes_classification_multi_level_feature.npy', {'gts': gts, 'preds': preds})
    
            


    
    
            





    
    


        




