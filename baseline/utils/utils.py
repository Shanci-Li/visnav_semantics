"""
Time:     2020/11/30 下午5:02
Author:   Ding Cheng(Deeachain)
File:     utils.py
Describe: Write during my study in Nanjing University of Information and Secience Technology
Github:   https://github.com/Deeachain
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from skimage import io
from mpl_toolkits.axes_grid1 import make_axes_locatable


def __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                  **kwargs):
    for name, m in feature.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d)):
            conv_init(m.weight, **kwargs)
        elif isinstance(m, norm_layer):
            m.eps = bn_eps
            m.momentum = bn_momentum
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


def init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                **kwargs):
    if isinstance(module_list, list):
        for feature in module_list:
            __init_weight(feature, conv_init, norm_layer, bn_eps, bn_momentum,
                          **kwargs)
    else:
        __init_weight(module_list, conv_init, norm_layer, bn_eps, bn_momentum,
                      **kwargs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_predict(args, output, gt, img_name, save_path):

    image = os.path.join(args.img_path, 'rgb/{:s}.png'.format(img_name))
    raw_image = io.imread(image)[:, :, :3]

    fig, axes = plt.subplots(1, 3)
    axes[0].axis('off')
    axes[0].imshow(raw_image)
    axes[0].set_title("Image")

    axes[1].axis('off')
    axes[1].imshow(output)
    axes[1].set_title("Prediction")

    axes[2].axis('off')
    im = axes[2].imshow(gt)
    axes[2].set_title("Ground Truth")

    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.savefig(os.path.join(save_path, img_name + '.png'), bbox_inches='tight', dpi=400)


def netParams(model):
    """
    computing total network parameters
    args:
       model: model
    return: the number of parameters
    """
    total_paramters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_paramters += p

    return total_paramters
