import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply, load_images_as_tensor_from_list
from pi3.utils.geometry import depth_edge
from pi3.models.pi3_classification import Pi3
import open3d as o3d
import numpy as np
import utils3d
import os
import matplotlib.pyplot as plt
import math
from pi3.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from pi3.utils.geometry_torch import recover_focal_shift
from pi3.utils.helper import create_pixel_coordinate_grid

def class_to_binary(pred_class, N):
    out = torch.ones(N, dtype=torch.int64)
    if pred_class < N:
        out[pred_class] = 0
    return out

if __name__ == '__main__':
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run inference with the Pi3 model.")
    
    parser.add_argument("--data_path", type=str, default='examples/skating.mp4',
                        help="Path to the input image directory or a video file.")
    parser.add_argument("--save_path", type=str, default='examples/result.ply',
                        help="Path to save the output .ply file.")
    parser.add_argument("--interval", type=int, default=-1,
                        help="Interval to sample image. Default: 1 for images dir, 10 for video")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to the model checkpoint file. Default: None")
    parser.add_argument("--device", type=str, default='cuda',
                        help="Device to run inference on ('cuda' or 'cpu'). Default: 'cuda'")
    parser.add_argument("--show_conf", action='store_true', default=False,
                        help="Whether to show the confidence maps.")
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
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


    # device = torch.device('cuda')
    # dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    # model = Pi3().to(device).eval()
    # from safetensors.torch import load_file
    # weight = load_file('ckpts/model.safetensors')
    # model.load_state_dict(weight)

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    # imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)
    image_root = '/home/disk3_SSD/ylx/dataset_pi3_classification/desk/input'
    image_name = ['0001.jpg','0002.jpg', '0003.jpg', '0004.jpg', '0005.jpg', '0006.jpg','0007.jpg', '0027.jpg']

    image_list = [os.path.join(image_root, name) for name in image_name]
    imgs = load_images_as_tensor_from_list(image_list=image_list, interval=args.interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension
        
    if args.show_conf:
        # imgs: (N, 3, H, W)
        imgs_show = imgs.detach().cpu().numpy()

        # conf: (N, H, W) logits
        conf = res['conf'][0].detach().cpu().numpy().squeeze(-1)
        conf = 1 / (1 + np.exp(-conf))   # sigmoid

        print('camera_poses', res['camera_poses'][0].cpu().numpy())

        N = imgs_show.shape[0]
        pairs_per_row = 5
        cols = pairs_per_row * 2
        rows = math.ceil(N / pairs_per_row)

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axs = np.array(axs).reshape(rows, cols)

        for i in range(N):
            row = i // pairs_per_row
            col = (i % pairs_per_row) * 2

            # ---- image ----
            img = imgs_show[i].transpose(1, 2, 0)  # (H, W, 3)
            img = np.clip(img, 0, 1)           # 如果是 0~1

            axs[row, col].imshow(img)
            axs[row, col].set_title(f'Img {i}')
            axs[row, col].axis('off')

            # ---- conf ----
            axs[row, col + 1].imshow(conf[i], cmap='jet')
            axs[row, col + 1].set_title(f'Conf {i}')
            axs[row, col + 1].axis('off')

        # 关掉多余格子
        for j in range(N * 2, rows * cols):
            axs.flat[j].axis('off')

        plt.tight_layout()
        plt.show()
        quit()

    print(res['logits'][0].shape)
    print(res['logits'][0])
    pred_class = res['logits'][0].argmax(dim=0).item()
    print("Predicted classes (per image + none): ", pred_class)
    binary = class_to_binary(pred_class, res['logits'][0].shape[0] - 1)
    print(binary)
    quit()



    # 4. process mask
    masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
    masks = torch.logical_and(masks, non_edge)[0]

    # 5. Save points
    print(f"Saving point cloud to: {args.save_path}")
    # write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], args.save_path)

    stride = 8  # 每隔4个像素取一个 downsampled mask
    mask_ds = torch.zeros_like(masks)
    mask_ds[..., ::stride, ::stride] = masks[..., ::stride, ::stride]
    write_ply(res['points'][0][mask_ds].cpu(), imgs.permute(0, 2, 3, 1)[mask_ds].cpu(), args.save_path)

    points = res['points'][0][mask_ds].cpu().numpy()
    colors = imgs.permute(0, 2, 3, 1)[mask_ds].cpu().numpy()
    extrinsics = torch.inverse(res['camera_poses'][0]).cpu().numpy()

    points_local = res["local_points"]
    masks = torch.sigmoid(res["conf"][..., 0]) > 0.1
    original_height, original_width = points_local.shape[-3:-1]
    aspect_ratio = original_width / original_height


    # (S, H, W, 3), with x, y coordinates and frame indices
    _, num_frames, height, width, _ = points_local.shape
    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)
    # use recover_focal_shift function from MoGe
    focal, shift = recover_focal_shift(points_local, masks)
    fx, fy = focal / 2 * (1 + aspect_ratio ** 2) ** 0.5 / aspect_ratio, focal / 2 * (1 + aspect_ratio ** 2) ** 0.5
    diag = (height**2 + width**2) ** 0.5
    fx_px = fx * (0.5 * diag)
    fy_px = fy * (0.5 * diag)
    cx_px = width * 0.5  
    cy_px = height * 0.5
    intrinsics = utils3d.torch.intrinsics_from_focal_center(fx_px, fy_px, cx_px, cy_px)
    intrinsics = intrinsics.cpu().numpy().squeeze(0)

    print(points_local.shape, points.shape, colors.shape, intrinsics.shape, extrinsics.shape)

    print("Converting to COLMAP format")
    points_3d = points
    points_rgb = colors * 255
    extrinsic = extrinsics
    intrinsic = intrinsics
    points_xyf = points_xyf[mask_ds.cpu().numpy()]
    print(points_3d.shape, points_xyf.shape, points_rgb.shape, extrinsic.shape, intrinsic.shape)

    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        np.array([height, width]),
        shared_camera=False,
        camera_type="SIMPLE_PINHOLE",
    )

    # print(f"Saving reconstruction to {args.data_path}/../sparse")
    # sparse_reconstruction_dir = os.path.join(args.data_path, "../sparse")
    # os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    # reconstruction.write(sparse_reconstruction_dir)

    # sparse_reconstruction_dir = '/home/disk3_SSD/ylx/dataset_pi3/pi3_visymscenes_5'
    # os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    # reconstruction.write(sparse_reconstruction_dir)
    

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # points_ds = np.asarray(pcd.points)
    # colors_ds = np.asarray(pcd.colors)
    # write_ply(points_ds, colors_ds, args.save_path)

    print("Done.")