import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import math


def get_ray_directions(H, W, focal, center=None):
    x = np.arange(W, dtype=np.float32) + 0.5
    y = np.arange(H, dtype=np.float32) + 0.5
    x, y = np.meshgrid(x, y)
    pix_coords = np.stack([x, y], axis=-1).reshape(-1, 2)
    i, j = pix_coords[..., 0:1], pix_coords[..., 1:]

    cent = center if center is not None else [W / 2, H / 2]
    directions = np.concatenate(
        [
            (i - cent[0]) / focal[0],
            (j - cent[1]) / focal[1],
            np.ones_like(i),
        ],
        axis=-1,
    )
    ray_dirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)
    return torch.tensor(ray_dirs, dtype=torch.float32)


class BlenderDataset(Dataset):
    def __init__(self, datadir, split="train", downsample=1):

        self.root_dir = datadir
        self.split = split
        self.downsample = downsample

        self.blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        self.points3D = None
        self.points3D_color = None

        with open(
            os.path.join(self.root_dir, f"transforms_{self.split}.json"), "r"
        ) as f:
            meta = json.load(f)
        if "w" in meta and "h" in meta:
            W, H = int(meta["w"]), int(meta["h"])
        else:
            W, H = 800, 800

        self.img_wh = (int(W / self.downsample), int(H / self.downsample))
        w, h = self.img_wh

        focal = (
            0.5 * w / math.tan(0.5 * meta["camera_angle_x"])
        )  # scaled focal length

        self.fx, self.fy = focal, focal
        self.intrinsics = torch.tensor(
            [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]]
        )

        cam_ray_dirs = get_ray_directions(
            h, w, [self.intrinsics[0, 0], self.intrinsics[1, 1]]
        )

        self.poses = []
        self.cameras = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_alphas = []
        for i, frame in enumerate(meta["frames"]):
            pose = np.array(frame["transform_matrix"]) @ self.blender2opencv
            c2w = torch.FloatTensor(pose)
            self.poses.append(c2w)
            world_ray_dirs = torch.einsum(
                "ij,kj->ik",
                cam_ray_dirs,
                c2w[:3, :3],
            )
            world_ray_origins = c2w[:3, 3] + torch.zeros_like(cam_ray_dirs)
            world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
            world_rays = world_rays.reshape(self.img_wh[1], self.img_wh[0], 6)

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            if self.downsample != 1.0:
                img = img.resize(self.img_wh, Image.LANCZOS)
            img = img.convert("RGBA")
            rgbas = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
            rgbs = rgbas[..., :3] * rgbas[..., 3:4] + (
                1 - rgbas[..., 3:4]
            )  # white bg
            img.close()

            self.all_rays.append(world_rays)
            self.all_rgbs.append(rgbs)
            self.all_alphas.append(rgbas[..., -1:])

        self.poses = torch.stack(self.poses)
        self.all_rays = torch.stack(self.all_rays)
        self.all_rgbs = torch.stack(self.all_rgbs)
        self.all_alphas = torch.stack(self.all_alphas)

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        if self.split == "train":  # use data in the buffers
            sample = {
                "rays": self.all_rays[idx],
                "rgbs": self.all_rgbs[idx],
                "alphas": self.all_alphas[idx],
            }

        else:  # create data for each image separately

            img = self.all_rgbs[idx]
            rays = self.all_rays[idx]
            alphas = self.all_alphas[idx]

            sample = {"rays": rays, "rgbs": img, "alphas": alphas}
        return sample


if __name__ == "__main__":
    pass
