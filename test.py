from datasets.visymscenes_dataset import VisymScenesDataset
import numpy as np
import pickle

# dopp_pair_path = '/home/disk8/dopp_data/pairs_metadata/train_pairs_visym.npy'
# dopp_pair = np.load(dopp_pair_path, allow_pickle=True)
# # sample 100 pairs
# # ...existing code...
# # 新增：从 dopp_pair 中不重复随机抽取最多 100 个样本并保存到文件
# dopp_pair
# rng = np.random.default_rng(42)  # 固定随机种子，方便复现；如不需要可去掉或改为 None
# n_samples = 100
# n_total = len(dopp_pair)

# if n_total <= n_samples:
#     sampled_pairs = dopp_pair.copy()
# else:
#     indices = rng.choice(n_total, size=n_samples, replace=False)
#     sampled_pairs = dopp_pair[indices]

# print(sampled_pairs)


# dataset = VisymScenesDataset(resolution=[518, 336], data_root='/home/disk8/dopp_data/visymscenes', mode='train', verbose=True)

all_pairs = []
with open("pair_data/neg_pairs_with_intrinsics.pkl", "rb") as f:
    all_pairs = pickle.load(f)

for idx, img in enumerate(all_pairs[0][4]):
    print(img)
    print(all_pairs[0][3][idx])
# print(all_pairs[0])
# print(len(all_pairs[0][3]))


# num_total = len(all_pairs)
# split_idx = int(0.8 * num_total)

# test_pairs = all_pairs[split_idx:]
# print(f'Total number of pairs: {len(all_pairs)}')
# print(f'Number of test pairs: {len(test_pairs)}')
# pos_pair = [all_pairs[37907]]
# with open("pair_data/neg_pairs_with_intrinsics.pkl", "wb") as f:
#     pickle.dump(pos_pair, f)


# print(len(all_pairs))
# num = 0
# for idx, pair in enumerate(all_pairs):
#     if pair[2] == '0':
#         print(pair)
#         print(idx)
#         input()
# print(f'Number of negative pairs: {num}, total pairs: {len(all_pairs)}')