import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import io
import pdb
import time
import argparse
from typing import Tuple, Union


def apply_offset(pc: np.ndarray, offset: np.ndarray, scale: float = 1.0, nodata_value: float = None) -> np.ndarray:
    """
    Apply offset and scaling to the point cloud data.
    @param pc:              [N, X] point cloud. (X >= 3, the fourth and later columns are non-coordinates).
    @param offset:          [3] offset vector.
    @param scale:           Float number for scaling.
    @param nodata_value:    Float number for nodata value.
    return                  Point cloud w/ offset and scaling.
    """
    pc = pc.copy()
    if nodata_value is None:
        pc[:, :3] -= offset.reshape(1, 3)
        pc[:, :3] /= scale
    else:
        nodata_value = float(nodata_value)
        pc[pc[:, 0] != nodata_value, :3] -= offset.reshape(1, 3)
        pc[pc[:, 0] != nodata_value, :3] /= scale
    return pc


def pc_local_search(big_pc: torch.Tensor, ref_patch: torch.Tensor, nodata_value: float = -1.0) -> torch.Tensor:
    """
    Point cloud local search based on XYZ boundaries.
    @param big_pc:          [N, D] point cloud. (D >= 3, the fourth and later columns are non-coordinates).
    @param ref_patch:       [K, 3] point cloud as reference.
    @param nodata_value:    Float number for nodata value.
    return                  A subset of pertinant point clouds.
    """
    if ref_patch.numel() == torch.sum(ref_patch == nodata_value).item():
        selected_pc = torch.empty(0)
    else:
        xyz_max, _ = ref_patch[ref_patch[:, 1] != nodata_value].max(dim=0)  # [3]
        xyz_min, _ = ref_patch[ref_patch[:, 1] != nodata_value].min(dim=0)  # [3]

        flag = torch.logical_and(big_pc[:, :3] <= xyz_max, big_pc[:, :3] >= xyz_min).sum(dim=1) == 3
        selected_pc = big_pc[flag]

        if len(selected_pc) == 0:
            selected_pc = torch.empty(0)
    
    return selected_pc


def split_scene_coord(sc: np.ndarray, block_h: int, block_w: int) -> np.ndarray:
    """
    Split the scene coordinate associated with image pixels.
    @param sc:          [H, W, 3] scene coordinates.
    @param block_h:     Block size in height direction.
    @param block_w:     Block size in width direction.
    return              an array of block-wise coordinates, [rows, cols, block_h, block_w, 3].
    """
    h, w = sc.shape[:2]

    assert h // block_h == h / block_h
    assert w // block_w == w / block_w

    sc_split_h_ls = np.vsplit(sc, np.arange(h)[::block_h][1:])  # vertical split in height direction

    sc_split = [[] for _ in range(len(sc_split_h_ls))]
    for row, sc_split_h in enumerate(sc_split_h_ls):
        sc_split_w = np.hsplit(sc_split_h, np.arange(w)[::block_w][1:])  # horizontal split in width direction
        sc_split[row] = sc_split_w

    return np.array(sc_split)


def convert_to_tensor(data: np.ndarray, cuda=False, retain_tensor=False, float16=False) -> Tuple[bool, Union[None, torch.Tensor]]:
    """
    Try making tensor from numpy array.
    """
    if float16:
        data_tensor = torch.tensor(data).bfloat16()
    else:
        data_tensor = torch.tensor(data).float()
    flag_ok = torch.isnan(data_tensor).sum() == 0 and torch.isinf(data_tensor).sum() == 0
    data_tensor = data_tensor if retain_tensor else torch.zeros(1)

    if flag_ok:
        data_tensor = data_tensor.cuda() if cuda else data_tensor
        return True, data_tensor
    else:
        del data_tensor
        return False, None


def sc_query(sc: torch.Tensor, pc: torch.Tensor, nodata_value: float = -1.0) -> np.ndarray:
    """
    Query the scene coords' semantic labels in the given point cloud.
    @param sc:              [H, W, 3] scene coordinates.
    @param pc:              [N, 4] point cloud w/ semantic labels.
    @param nodata_value:    Float number for nodata value.
    @return                 [H, W] semantic label.
    """
    h, w = sc.shape[:2]

    pc = pc.clone()

    sc = sc.reshape(-1, 3)  # [K, 3]
    mask_nodata = sc[:, 0] == nodata_value

    sc_cdist = sc[torch.logical_not(mask_nodata)]  # [K', 3]
    pc_cdist = pc[:, :3]  # [N, 3]

    # torch cdist for distance computation, only p=2 is supported as of pytorch 1.9!
    # See issue: https://github.com/pytorch/pytorch/issues/49928
    qeury2pc_dist = torch.cdist(sc_cdist, pc_cdist, p=2.0)  # [K', N]

    # matrix multiplication, too much GPU memory, don't use.
    # qeury2pc_dist = torch.mm(sc_cdist, pc_cdist.transpose(1, 0))  # [K', N]

    # l1 distance, too much GPU memory, don't use.
    # qeury2pc_dist = (sc_cdist[:, None, :] - pc_cdist[None, :, :]).abs().sum(dim=-1)  # [K', N]

    cloest_dist, idx_cloest_pt = qeury2pc_dist.min(dim=1)  # [K'] + [K']

    semantics_label = -torch.ones(h*w, 2).to(sc.device)  # [H * W]
    semantics_label[torch.logical_not(mask_nodata), 0] = pc[idx_cloest_pt, -1].float()
    semantics_label[torch.logical_not(mask_nodata), 1] = cloest_dist.float()

    semantics_label = semantics_label.reshape(h, w, 2).cpu().numpy()

    return semantics_label


def check_mem(sc_cdist_len: int, pc_cdist_len: int, max_mem_GB: int = 32) -> bool:
    '''
    check whether the cdist operation will out of memory
    :param sc_cdist_len: number of pixels in the split image patch
    :param pc_cdist_len: number of point in the query scene
    :param max_mem_GB: max free GPU memory in GB
    :return: bool
    '''
    if ((sc_cdist_len * pc_cdist_len) / 1e9) <= (max_mem_GB * 4 / 15):
        return True
    else:
        return False


def find_opt_split(_sc: np.ndarray, _big_pc_label_tensor: torch.Tensor,
                   block_h: int, block_w: int, max_mem_GB: int = 32) -> (int, int):
    '''
    find the optimal strategy to split the input image while fully utilise the GPU
    :param max_mem_GB:            max GPU free memory in GB
    :param block_w:               default split block width
    :param block_h:               default split block heiggt
    :param _sc:                   input image [480, 720, 3]
    :param _big_pc_label_tensor:  entire point cloud w/ label
    :return block_h, block_w:     optimal split of the image in [block_h, block_w]
    '''

    sc_cdist_len, pc_cdist_len = block_h * block_w, _big_pc_label_tensor.shape[0]
    pattern_idx = 0
    optional_list = [(240, 180), (120, 180), (120, 90), (60, 90), (60, 72), (60, 45), (48, 45),
                     (48, 36), (24, 24), (24, 12), (12, 12), (6, 6), (1, 1)]
    while not check_mem(sc_cdist_len, pc_cdist_len, max_mem_GB):
        block_h, block_w = optional_list[pattern_idx]
        sc_cdist_len = block_h * block_w

        _sc_split_test = split_scene_coord(_sc, block_h, block_w)  # [rows, cols, b_h, b_w, 3]

        flag_tensor, _sc_split_test = convert_to_tensor(_sc_split_test, cuda=True, retain_tensor=True)
        assert flag_tensor
        # selected_pc_len_max
        pc_cdist_len = 0
        for row in range(_sc_split_test.shape[0]):
            for col in range(_sc_split_test.shape[1]):
                selected_pc = pc_local_search(_big_pc_label_tensor, _sc_split_test[row, col].reshape(-1, 3),
                                              nodata_value=-1)  # [X, 4]
                pc_cdist_len = max(pc_cdist_len, selected_pc.shape[0])
        pattern_idx += 1

    # release the GPU memory
    torch.cuda.empty_cache()

    return block_h, block_w


def main():

    args = config_parser()
    downsample_rate = args.downsample_rate
    max_mem_GB = 5

    # read point cloud with semantic label data from .npy file
    _big_pc_label = np.load("big_pc_label.npy")  # [X, 4]
    bound_xyz_min = _big_pc_label[:, :3].min(axis=0)  # [3]
    bound_xyz_max = _big_pc_label[:, :3].max(axis=0)  # [3]
    offset_center = (bound_xyz_max + bound_xyz_min) / 2  # [3]
    interval_xyz = bound_xyz_max - bound_xyz_min  # [3]
    if args.float16:
        scale = np.array(interval_xyz / 1.e5, np.float64)  # [3]
    else:
        scale = 1.0
    print('Offset origin XYZ: {}, {}, {}, scale: {}'.format(*offset_center, scale))
    _big_pc_label = apply_offset(_big_pc_label, offset_center, scale, nodata_value=None)  # [X, 4]

    flag_tensor, _big_pc_label_tensor = convert_to_tensor(_big_pc_label, cuda=True, retain_tensor=True, float16=args.float16)
    assert flag_tensor, "Cannot build tensor for the original data (w/ offset)!"

    file_ls = ['_'.join(item.split('_')[:-1]) for item in os.listdir('scene_coord')]
    file_ls = np.unique(file_ls).tolist()
    print(file_ls)

    for idx_dp, file_name in tqdm(enumerate(file_ls)):

        time_start = time.time()
        block_h, block_w = args.block_h, args.block_w
        """Load ray-traced point cloud"""
        _sc = np.load('scene_coord/{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
        raw_image = io.imread('scene_coord/{:s}_img.png'.format(file_name))[:, :, :3]
        _sc = _sc[::downsample_rate, ::downsample_rate, :]  # [H, W, 3]
        _sc_raw_size = _sc.shape  # [H, W, 3]
        _sc = apply_offset(_sc.reshape(-1, 3), offset_center, scale, nodata_value=-1).reshape(_sc_raw_size)  # [H, W, 3], numpy array

        block_h, block_w = find_opt_split(_sc, _big_pc_label_tensor, block_h, block_w, max_mem_GB= 5)

        _sc_split = split_scene_coord(_sc, block_h, block_w)  # [rows, cols, b_h, b_w, 3]

        flag_tensor, _sc_split = convert_to_tensor(_sc_split, cuda=True, retain_tensor=True, float16=args.float16)
        assert flag_tensor

        """Recover the semantic labels (divide and conquer)"""
        semantics_label_ls = [[[] for _ in range(_sc_split.shape[1])] for _ in range(_sc_split.shape[0])]
        semantics_distance_ls = [[[] for _ in range(_sc_split.shape[1])] for _ in range(_sc_split.shape[0])]
        ttl_time_search, ttl_time_query = 0.0, 0.0
        for row in range(_sc_split.shape[0]):
            for col in range(_sc_split.shape[1]):
                time_search = time.time()
                selected_pc = pc_local_search(_big_pc_label_tensor, _sc_split[row, col].reshape(-1, 3),
                                              nodata_value=-1)  # [X, 4]
                ttl_time_search += time.time() - time_search

                time_query = time.time()
                if len(selected_pc):
                    semantic_label = sc_query(_sc_split[row, col], selected_pc, nodata_value=-1.0)
                else:
                    semantic_label = -np.ones_like(_sc_split[row, col].cpu().float().numpy())[:, :, :2]
                
                ttl_time_query += time.time() - time_query

                semantics_label_ls[row][col] = semantic_label[:, :, 0]
                semantics_distance_ls[row][col] = semantic_label[:, :, 1] * scale
                del selected_pc

        semantics_label = np.block(semantics_label_ls)  # [H, W]
        semantics_label[semantics_label == -1] = 0
        semantics_label = np.array(semantics_label, np.uint8)
        semantics_distance = np.block(semantics_distance_ls).astype(np.float32)  # [H, W]
        tiem_elapsed = time.time() - time_start
        print("Semantics label recovery time: {:.1f}s, unique labels: {}".format(tiem_elapsed, np.unique(semantics_label)))
        torch.cuda.empty_cache()

        """Results saving"""
        np.save('semantics_label_{:s}'.format(file_name), semantics_label)
        np.save('semantics_distance_{:s}'.format(file_name), semantics_distance)

        if args.plot:
            fig, axes = plt.subplots(1, 3)
            axes[0].axis('off')
            axes[0].imshow(raw_image)
            axes[0].set_title("Image")

            axes[1].axis('off')
            axes[1].imshow(semantics_label)
            axes[1].set_title("Semantics")

            axes[2].axis('off')
            im = axes[2].imshow(semantics_distance)
            axes[2].set_title("Cloest point distance")

            from mpl_toolkits.axes_grid1 import make_axes_locatable

            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            plt.colorbar(im, cax=cax)
            plt.savefig('semantics_label_{:s}.png'.format(file_name), bbox_inches='tight', dpi=400)
        

def config_parser():
    parser = argparse.ArgumentParser(
    description='Semantic label recovery script.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--float16', action='store_true',
                        help='Use float16 number to accelerate computation (unstable!)')

    parser.add_argument('--downsample_rate', type=int, default=1,
                    help='Downsampling rate.')

    parser.add_argument('--block_h', type=int, default=240,
                    help='Cell block height.')

    parser.add_argument('--block_w', type=int, default=360,
                    help='Cell block width.')

    parser.add_argument('--plot', action='store_true',
                    help='Plot visualized results.')

    opt = parser.parse_args()

    if opt.float16:
        print("Warning: float16 mode is highly unstable!")
    return opt

if __name__ == '__main__':
    main()
