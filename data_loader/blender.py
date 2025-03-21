import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
import json
import math


from .ray_utils import *


class BlenderDataset(Dataset):
    def __init__(
        self, datadir, split="train", downsample=1, is_stack=False, N_vis=-1
    ):
        assert downsample in [1, 2, 4, 8]

        self.root_dir = datadir
        self.split = split
        self.is_stack = is_stack
        self.define_transforms()

        self.points3D = None
        self.points3D_color = None

        with open(os.path.join(self.root_dir, f"transforms_{self.split}.json"), 'r') as f:
            meta = json.load(f)
        if 'w' in meta and 'h' in meta:
            W, H = int(meta['w']), int(meta['h'])
        else:
            W, H = 800, 800

        self.w, self.h = W, H

        if downsample > 1:
            w, h = w // downsample, h // downsample
        
        self.img_wh = (self.w, self.h)

        self.near, self.far = 2.0, 6.0

        self.focal = 0.5 * self.w / math.tan(0.5 * meta['camera_angle_x']) # scaled focal length

        fx, fy, cx, cy = self.focal, self.focal, self.w // 2, self.h // 2
        self.intrinsics = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        # type_ = cam.camera_type

        # if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        #     params = None
        #     camtype = "perspective"

        # elif type_ == 1 or type_ == "PINHOLE":
        #     params = None
        #     camtype = "perspective"

        # else:
        #     assert type_ > 1, "Only support pinhole camera model."

        self.directions = get_ray_directions(
            self.h, self.w, [self.intrinsics[0, 0], self.intrinsics[1, 1]])  # (h, w, 3)

        self.directions = self.directions / torch.norm(
            self.directions, dim=-1, keepdim=True
        )
        # directions normalization

        self.poses = []
        self.cameras = []
        self.all_rays = []
        self.all_rgbs = []
        self.all_alphas = []
        for i, frame in enumerate(meta['frames']):
            c2w = torch.from_numpy(np.array(frame['transform_matrix'])[:3, :4]).to(torch.float32)
            # use float32 in radfoam
            c2w[:3, 1:3] *= -1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            self.poses.append(c2w)

            img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
            img = Image.open(img_path)
            img = self.transform(img) # (h*w, 4) RGB
            img = img.permute(1, 2, 0)
            image_mask = img[:, :, 3]
            img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4])
            # by default white background
            img = img.view(-1, 3)
            image_mask = image_mask.view(-1, 1)

            self.all_rgbs += [img]
            self.all_alphas += [image_mask]

            rays_o, rays_d = get_rays(self.directions, c2w)
            self.all_rays += [torch.cat([rays_o, rays_d], -1)]  # (h*w, 6)
            self.cameras += [rays_o.view(-1, 3)[0]]

        self.poses = torch.stack(self.poses)
        self.cameras = torch.stack(self.cameras)
        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0)
            self.all_rgbs = torch.cat(self.all_rgbs, 0)
            self.all_alphas = torch.cat(self.all_alphas, 0)

        else:
            self.all_rays = torch.stack(self.all_rays, 0)
            assert (
                self.all_rays[0].shape[-1] == 6
            ), "Rays should have 6 as last dim"
            self.all_rays = self.all_rays.reshape(-1, *self.img_wh[::-1], 6)

            self.all_rgbs = torch.stack(self.all_rgbs, 0)
            assert (
                self.all_rgbs[0].shape[-1] == 3
            ), "RGBs should have 3 as last dim"
            self.all_rgbs = self.all_rgbs.reshape(-1, *self.img_wh[::-1], 3)

            self.all_alphas = torch.stack(self.all_alphas, 0)
            assert (
                self.all_alphas[0].shape[-1] == 1
            ), "Alphas should have 1 as last dim"
            self.all_alphas = self.all_alphas.reshape(-1, *self.img_wh[::-1], 1)

    def define_transforms(self):
        self.transform = T.ToTensor()

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
