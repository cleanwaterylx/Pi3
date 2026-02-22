import numpy as np
import re
from collections import defaultdict
import networkx as nx
from community import community_louvain
import random
from tqdm import tqdm
import torch
from pi3.utils.basic import load_images_as_tensor_from_list
from pi3.models.pi3_classification import Pi3

img_name2id = {}
name2idx = {}
idx2name = {}
random.seed(42)

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

def parse_image_map_file(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            img_id, image_name = line.split(' ')
            img_name2id.update({image_name.strip(): int(img_id)})

def parse_image_pair_file(file_path):
    # 用于存储边信息
    edges = []
    G = nx.Graph()

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            img1, img2 = parts[2], parts[3]
            inliers = int(parts[4])
            matches = int(parts[5])

            # quaternion 部分在 parts[5:11]，里面有 "+"，需要过滤
            q_tokens = [p for p in parts[6:13] if p != '+']
            if len(q_tokens) != 4:
                raise ValueError(f"无法解析 quaternion: {parts[6:12]}")

            # 去掉 i,j,k
            def clean_token(tok):
                return float(re.sub(r'[ijk]', '', tok))

            qx, qy, qz, qw = map(clean_token, q_tokens)
            # 顺序 (x, y, z, w)
            q = np.array([qx, qy, qz, qw])

            # 平移向量
            tx, ty, tz = map(float, parts[13:16])
            t = np.array([tx, ty, tz])

            edges.append((img1, img2, inliers, inliers / matches, q, t))
            G.add_edge(img1, img2, weight=inliers)
    print("Graph info: nodes edges", G.number_of_nodes(), G.number_of_edges())
    return G

def random_simple_path(G, start, L):
    path = [start]
    visited = {start}
    cur = start

    for _ in range(L - 1):
        nbrs = [n for n in sorted(G.neighbors(cur)) if n not in visited]
        if not nbrs:
            break
        cur = random.choice(nbrs)
        path.append(cur)
        visited.add(cur)

    return path


if __name__ == '__main__':
    name = '/home/disk3_SSD/ylx/dataset_pi3_classification/26'
    parse_image_map_file(f'{name}/sparse/image_map.txt')
    # input()
    G = parse_image_pair_file(f'{name}/sparse/image_pair_inliers_relpose_final.txt')
    communitys = community_louvain.best_partition(G, random_state=42)

    clusters = defaultdict(list)
    for node, cid in communitys.items():
        clusters[cid].append(node)
    clusters = dict(clusters)
    clusters_sorted = sorted(
        clusters.values(),
        key=lambda x: len(x),
        reverse=True
    )
    for idx, c in enumerate(clusters_sorted):
        print(f'Cluster {idx}, size: {len(c)}', sorted(c))

    # Prepare model
    print(f"Loading model...")
    device = torch.device('cuda')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = Pi3().to(device).eval()
    from safetensors.torch import load_file
    weight = load_file('ckpts/model_pi3_visymscenes_classification.safetensors')
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

    # use pi3 model to classify
    for idx, cluster in enumerate(clusters_sorted):
        # sample in path in view graph and batched processing
        sample_num = 5
        cluster = sorted(cluster)
        if len(cluster) < sample_num + 1:
            continue
        subgraph = G.subgraph(cluster).copy()  # todo wether to use copy
        for _ in tqdm(range(5)):
            batch_image_lists = []
            for _ in range(32):  # generate 32 samples path per cluster
                # sample_imgs = random.sample(cluster, sample_num)
                sample_imgs = random_simple_path(subgraph, random.choice(cluster), sample_num)
                image_lists = []
                for i, img_name in enumerate(sample_imgs):
                    image_lists.append(f'{name}/input/{img_name}')
                batch_image_lists.append(image_lists)

            # process batch_image_lists with pi3 model to classify
            batch_size = len(batch_image_lists)
            # flatten the list of lists
            flat_image_list = [img for sublist in batch_image_lists for img in sublist]
            # load images as tensor
            images_tensor = load_images_as_tensor_from_list(flat_image_list)
            images_tensor = images_tensor.to(device)  # [B*N, 3, H, W]
            B = batch_size
            N = sample_num
            images_tensor = images_tensor.view(B, N, 3, images_tensor.shape[2], images_tensor.shape[3])  # [B, N, 3, H, W]
            with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=dtype):
                        res = model(images_tensor)
            # print(res['logits'].shape)
            pred_class = res['logits'].argmax(dim=1).cpu().numpy()
            for i, image_lists in enumerate(batch_image_lists):
                outlier_idx = pred_class[i]
                # print(outlier_idx)
                if outlier_idx < N:
                    # print(image_lists[outlier_idx])
                    outlier = image_lists[outlier_idx].split('/')[-1]
                    pairs = [
                        (img.split('/')[-1], outlier)
                        for i, img in enumerate(image_lists)
                        if i != outlier_idx
                    ]
                    for u, v in pairs:
                        if G.has_edge(u, v):
                            G[u][v]["weight"] *= 0.5  # reduce weight
                # else:
                #     print(image_lists)
                #     print("All good")
                

    communitys = community_louvain.best_partition(G, random_state=42)
    clusters = defaultdict(list)
    for node, cid in communitys.items():
        clusters[cid].append(node)
    clusters = dict(clusters)
    clusters_sorted = sorted(
        clusters.values(),
        key=lambda x: len(x),
        reverse=True
    )
    for idx, c in enumerate(clusters_sorted):
        print(f'Cluster {idx}, size: {len(c)}', sorted(c))
    input()

    

    with open(f'{name}/image_clusters_louvain.txt', 'w') as f:
        for idx, c in enumerate(clusters_sorted):
            f.write(f'# Cluster {idx}, size: {len(c)}\n')
            c = sorted(c)
            for img in c:
                f.write(f'{img} ')
            f.write('\n')