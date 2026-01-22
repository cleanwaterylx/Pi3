import torch
import argparse
from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import open3d as o3d
import numpy as np
import utils3d
import os
from pi3.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track
from pi3.utils.geometry_torch import recover_focal_shift
from pi3.utils.helper import create_pixel_coordinate_grid

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
                        
    args = parser.parse_args()
    if args.interval < 0:
        args.interval = 10 if args.data_path.endswith('.mp4') else 1
    print(f'Sampling interval: {args.interval}')

    # from pi3.utils.debug import setup_debug
    # setup_debug()

    # 1. Prepare model
    print(f"Loading model...")
    device = torch.device(args.device)
    if args.ckpt is not None:
        model = Pi3().to(device).eval()
        if args.ckpt.endswith('.safetensors'):
            from safetensors.torch import load_file
            weight = load_file(args.ckpt)
        else:
            weight = torch.load(args.ckpt, map_location=device, weights_only=False)
        
        model.load_state_dict(weight)
    else:
        model = Pi3.from_pretrained("yyfz233/Pi3").to(device).eval()
        # or download checkpoints from `https://huggingface.co/yyfz233/Pi3/resolve/main/model.safetensors`, and `--ckpt ckpts/model.safetensors`

    # 2. Prepare input data
    # The load_images_as_tensor function will print the loading path
    imgs = load_images_as_tensor(args.data_path, interval=args.interval).to(device) # (N, 3, H, W)

    # 3. Infer
    print("Running model inference...")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            res = model(imgs[None]) # Add batch dimension

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

    print(f"Saving reconstruction to {args.data_path}/../sparse")
    sparse_reconstruction_dir = os.path.join(args.data_path, "../sparse")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    

    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # pcd.colors = o3d.utility.Vector3dVector(colors)

    # pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # points_ds = np.asarray(pcd.points)
    # colors_ds = np.asarray(pcd.colors)
    # write_ply(points_ds, colors_ds, args.save_path)

    print("Done.")