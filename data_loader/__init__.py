import os

import numpy as np
import einops
import torch

import radfoam

from .colmap import COLMAPDataset
from .blender import BlenderDataset


dataset_dict = {
    "colmap": COLMAPDataset,
    "blender": BlenderDataset,
}


def get_up(c2ws):
    right = c2ws[:, :3, 0]
    down = c2ws[:, :3, 1]
    forward = c2ws[:, :3, 2]

    A = torch.einsum("bi,bj->bij", right, right).sum(dim=0)
    A += torch.einsum("bi,bj->bij", forward, forward).sum(dim=0) * 0.02

    l, V = torch.linalg.eig(A)

    min_idx = torch.argmin(l.real)
    global_up = V[:, min_idx].real
    global_up *= torch.einsum("bi,i->b", -down, global_up).sum().sign()

    return global_up


class DataHandler:
    def __init__(self, dataset_args, rays_per_batch, device="cuda"):
        self.args = dataset_args
        self.rays_per_batch = rays_per_batch
        self.device = torch.device(device)
        self.img_wh = None
        self.patch_size = 8

    def reload(self, split, downsample=None):
        data_dir = os.path.join(self.args.data_path, self.args.scene)
        dataset = dataset_dict[self.args.dataset]
        if downsample is not None:
            split_dataset = dataset(
                data_dir, split=split, downsample=downsample
            )
        else:
            split_dataset = dataset(data_dir, split=split)
        self.img_wh = split_dataset.img_wh
        self.fx = split_dataset.fx
        self.fy = split_dataset.fy
        self.c2ws = split_dataset.poses
        self.rays, self.rgbs = split_dataset.all_rays, split_dataset.all_rgbs
        self.alphas = getattr(
            split_dataset, "all_alphas", torch.ones_like(self.rgbs[..., 0:1])
        )

        self.viewer_up = get_up(self.c2ws)
        self.viewer_pos = self.c2ws[0, :3, 3]
        self.viewer_forward = self.c2ws[0, :3, 2]

        try:
            self.points3D = split_dataset.points3D
            self.points3D_colors = split_dataset.points3D_color
        except:
            self.points3D = None
            self.points3D_colors = None

        if split == "train":
            if self.args.patch_based:
                dw = self.img_wh[0] - (self.img_wh[0] % self.patch_size)
                dh = self.img_wh[1] - (self.img_wh[1] % self.patch_size)
                w_inds = np.linspace(0, self.img_wh[0] - 1, dw, dtype=int)
                h_inds = np.linspace(0, self.img_wh[1] - 1, dh, dtype=int)

                self.train_rays = self.rays[:, h_inds, :, :]
                self.train_rays = self.train_rays[:, :, w_inds, :]
                self.train_rgbs = self.rgbs[:, h_inds, :, :]
                self.train_rgbs = self.train_rgbs[:, :, w_inds, :]

                self.train_rays = einops.rearrange(
                    self.train_rays,
                    "n (x ph) (y pw) r -> (n x y) ph pw r",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )
                self.train_rgbs = einops.rearrange(
                    self.train_rgbs,
                    "n (x ph) (y pw) c -> (n x y) ph pw c",
                    ph=self.patch_size,
                    pw=self.patch_size,
                )

                self.batch_size = self.rays_per_batch // (self.patch_size**2)
            else:
                self.train_rays = einops.rearrange(
                    self.rays, "n h w r -> (n h w) r"
                )
                self.train_rgbs = einops.rearrange(
                    self.rgbs, "n h w c -> (n h w) c"
                )
                self.train_alphas = einops.rearrange(
                    self.alphas, "n h w 1 -> (n h w) 1"
                )

                self.batch_size = self.rays_per_batch

    def get_iter(self):
        ray_batch_fetcher = radfoam.BatchFetcher(
            self.train_rays, self.batch_size, shuffle=True
        )
        rgb_batch_fetcher = radfoam.BatchFetcher(
            self.train_rgbs, self.batch_size, shuffle=True
        )
        alpha_batch_fetcher = radfoam.BatchFetcher(
            self.train_alphas, self.batch_size, shuffle=True
        )

        while True:
            ray_batch = ray_batch_fetcher.next()
            rgb_batch = rgb_batch_fetcher.next()
            alpha_batch = alpha_batch_fetcher.next()

            yield ray_batch, rgb_batch, alpha_batch
