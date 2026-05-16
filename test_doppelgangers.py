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



if __name__ == '__main__':
    rng = np.random.default_rng(42)
    data_root='/home/disk8/dopp_data/'
    dopp_pair = np.load('/home/disk8/dopp_data/pairs_metadata/test_train_pairs.npy', allow_pickle=True)
    # shuffle the pairs
    np.random.seed(42) 
    np.random.shuffle(dopp_pair)
    print(f'Number of test DoppPairs: {len(dopp_pair)}')

    
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
        
    # 2. Prepare model
    print(f"Loading model...")
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = Pi3().to(device).eval()
    from safetensors.torch import load_file
    weight = load_file('ckpts/pi3_visymscenes_feature_align_first_img_lora_visym_dopp_epoch2.safetensors')
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
    
    gts = []
    preds = []
    logits = []
    skipped_pairs = []
    all_true = 0
    not_all_true = 0
    for pair in tqdm(dopp_pair):
        image_0_relative_path, image_1_relative_path, pos_neg_pair_label = pair
        image_0_name = os.path.join(data_root, image_0_relative_path)
        image_1_name = os.path.join(data_root, image_1_relative_path)
        selected_imgs = [image_0_name, image_1_name]
        images_tensor = load_images_as_tensor_from_list(selected_imgs)
        if images_tensor.ndim != 4 or images_tensor.shape[0] != len(selected_imgs):
            skipped_pairs.append((image_0_relative_path, image_1_relative_path, pos_neg_pair_label))
            tqdm.write(f"Skipping pair due to image load/process failure: {selected_imgs}")
            continue

        images_tensor = images_tensor.to(device)
        labels = [1, pos_neg_pair_label]
        
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
        # print(sim)
        # print(labels)
        # input()

        gts.append(int(pos_neg_pair_label))
        preds.append(int(sim[0, -1, 0].detach().cpu().item()))
        print(int(pos_neg_pair_label), sim[0, -1, 0].detach().cpu().item())
        input()
        
            
    # np.save('gts_preds_visym_test_pi3_visymscenes_feature_align_first_img_512_epoch4_lora.npy', {'gts': gts, 'preds': preds})    
    print(f"Skipped pairs due to image load/process failure: {len(skipped_pairs)}")
    
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
    
    # np.save('gts_preds_visym_test_pi3_visymscenes_feature_align_first_img_512_epoch4_lora.npy', {'gts': gts, 'preds': preds})
    
        
        
        
        
        
        
        
        
    
    
    
