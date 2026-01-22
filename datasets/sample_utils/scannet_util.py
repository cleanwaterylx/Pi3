import os
import json
import numpy as np
from tqdm import tqdm


def load_pose(pose_path):
    """
    Load a 4x4 camera pose matrix from txt file
    """
    with open(pose_path, 'r') as f:
        pose_text = f.read()
    pose = np.array([float(x) for x in pose_text.split()], dtype=np.float32)
    pose = pose.reshape(4, 4)
    return pose


def is_invalid_pose(camera_pose):
    """
    Check whether camera pose contains NaN or Inf
    """
    return np.isinf(camera_pose).any() or np.isnan(camera_pose).any()


def generate_invalid_list(data_root):
    """
    Iterate over all ScanNet scenes and detect invalid camera poses
    """
    scenes = sorted(os.listdir(data_root))
    invalid_list = {}

    for scene in tqdm(scenes, desc="Processing scenes"):
        scene_path = os.path.join(data_root, scene)
        pose_dir = os.path.join(scene_path, 'pose')

        if not os.path.isdir(pose_dir):
            continue

        invalid_list[scene] = []

        pose_files = sorted(os.listdir(pose_dir))
        for pose_file in pose_files:
            if not pose_file.endswith('.txt'):
                continue

            frame_id = int(os.path.splitext(pose_file)[0])
            pose_path = os.path.join(pose_dir, pose_file)

            try:
                camera_pose = load_pose(pose_path)
            except Exception as e:
                # 文件损坏或无法解析，直接标记为 invalid
                invalid_list[scene].append(frame_id)
                continue

            if is_invalid_pose(camera_pose):
                invalid_list[scene].append(frame_id)

    return invalid_list


if __name__ == "__main__":
    # 修改为你的 ScanNet 数据根目录
    DATA_ROOT = "/home/disk8/scannet_part/scans"

    os.makedirs("data", exist_ok=True)

    invalid_list = generate_invalid_list(DATA_ROOT)

    output_path = "data/scannet_invalid_list.json"
    with open(output_path, "w") as f:
        json.dump(invalid_list, f, indent=2)

    print(f"Saved invalid list to {output_path}")
