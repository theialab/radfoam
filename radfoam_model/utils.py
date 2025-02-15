import numpy as np
import torch

from data_loader.ray_utils import *
import radfoam


def inverse_softplus(x, beta, scale=1):
    # log(exp(scale*x)-1)/scale
    out = x / scale
    mask = x * beta < 20 * scale
    out[mask] = torch.log(torch.exp(beta * out[mask]) - 1 + 1e-10) / beta
    return out


def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(-1, img1.shape[-1]).mean(0, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def get_expon_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=1_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0
        t = np.clip((step - warmup_steps) / (max_steps - warmup_steps), 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return log_lerp

    return helper


def get_cosine_lr_func(
    lr_init,
    lr_final,
    warmup_steps=0,
    max_steps=10_000,
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if warmup_steps and step < warmup_steps:
            return lr_init * step / warmup_steps
        elif step > max_steps:
            return 0.0
        lr_cos = lr_final + 0.5 * (lr_init - lr_final) * (
            1
            + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps))
        )
        return lr_cos

    return helper
