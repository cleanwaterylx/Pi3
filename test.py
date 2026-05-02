from datasets.visymscenes_dataset import VisymScenesDataset
import numpy as np
import pickle
import random
import torch
from tqdm import tqdm

data = np.load('gts_preds_visym_test_pi3_visymscenes_feature_align_first_img_512_epoch2.npy', allow_pickle=True).item()
wrong = 0
for i in range(len(data['gts'])):
    if data['gts'][i] != data['preds'][i]:
        wrong += 1
print(f"Total: {len(data['gts'])}, Wrong: {wrong}, Accuracy: {(len(data['gts'])-wrong)/len(data['gts']):.4f}")