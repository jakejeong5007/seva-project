# import json
# from pathlib import Path

# def inspect_scene(scene_path: str):
#     scene = Path(scene_path)
#     tf = json.load(open(scene / "transforms.json"))

#     print("\nSCENE:", scene)
#     print("num frames:", len(tf["frames"]))

#     file_paths = [f["file_path"] for f in tf["frames"]]
#     print("\nFirst 20 frame paths in transforms.json:")
#     for i, p in enumerate(file_paths[:20]):
#         print(f"{i:04d}: {p}")

#     sorted_paths = sorted(file_paths)
#     same_as_sorted = file_paths == sorted_paths
#     print("\ntransforms frame order == sorted(file_path)?", same_as_sorted)

#     if not same_as_sorted:
#         print("\nFirst 20 sorted paths:")
#         for i, p in enumerate(sorted_paths[:20]):
#             print(f"{i:04d}: {p}")

#     for split_file in sorted(scene.glob("train_test_split_*.json")):
#         split = json.load(open(split_file))
#         print("\nSPLIT:", split_file.name)
#         print("train_ids:", split["train_ids"])
#         print("test_ids first 20:", split["test_ids"][:20])

#         print("train image paths:")
#         for idx in split["train_ids"]:
#             print(f"  {idx:04d}: {file_paths[idx]}")

#         print("first 10 test image paths:")
#         for idx in split["test_ids"][:10]:
#             print(f"  {idx:04d}: {file_paths[idx]}")

# inspect_scene("/home/jake/projects/stable-virtual-camera/assets_demo_cli/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557")
# inspect_scene("/home/jake/projects/stable-virtual-camera/dataset/dl3dv_parsed/11K/scenes/000000")


# import json
# from pathlib import Path
# # from PIL import Image, ImageDraw

# scene = Path("/home/jake/projects/stable-virtual-camera/dataset/dl3dv_parsed/11K/scenes/000000")
# tf = json.load(open(scene / "transforms.json"))
# split = json.load(open(scene / "train_test_split_6.json"))

# for idx in split["train_ids"]:
#     p = scene / tf["frames"][idx]["file_path"]
#     print(idx, p)

import json
import numpy as np
from pathlib import Path

def camera_stats(scene_path: str):
    scene = Path(scene_path)
    tf = json.load(open(scene / "transforms.json"))

    c2ws = np.array([np.array(f["transform_matrix"], dtype=np.float64) for f in tf["frames"]])
    centers = c2ws[:, :3, 3]

    # In OpenGL-style camera coordinates, cameras typically look along local -Z.
    forward_gl = -c2ws[:, :3, 2]
    forward_cv =  c2ws[:, :3, 2]

    center_med = np.median(centers, axis=0)
    to_center = center_med[None] - centers
    to_center = to_center / (np.linalg.norm(to_center, axis=1, keepdims=True) + 1e-8)

    score_gl = np.mean(np.sum(forward_gl * to_center, axis=1))
    score_cv = np.mean(np.sum(forward_cv * to_center, axis=1))

    step = np.linalg.norm(np.diff(centers, axis=0), axis=1)

    print("\nSCENE:", scene)
    print("OpenGL -Z look-at score:", score_gl)
    print("OpenCV +Z look-at score:", score_cv)
    print("camera center min:", centers.min(axis=0))
    print("camera center max:", centers.max(axis=0))
    print("camera center span:", centers.max(axis=0) - centers.min(axis=0))
    print("median camera step:", np.median(step))
    print("max camera step:", np.max(step))

camera_stats("/home/jake/projects/stable-virtual-camera/assets_demo_cli/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557")
camera_stats("/home/jake/projects/stable-virtual-camera/dataset/dl3dv_parsed/11K/scenes/000000")

# inspect_scene("/home/jake/projects/stable-virtual-camera/assets_demo_cli/assets_demo_cli/dl3d140-165f5af8bfe32f70595a1c9393a6e442acf7af019998275144f605b89a306557")
# inspect_scene("/home/jake/projects/stable-virtual-camera/dataset/dl3dv_parsed/11K/scenes/000000")