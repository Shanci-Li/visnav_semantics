import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import io
import pdb
import pyproj
import time


def get_ecef_origin():
    """Shift the origin to make the value of coordinates in ECEF smaller and increase training stability"""
    # Warning: this is dataset specific!
    ori_lon, ori_lat, ori_alt = 6.5668, 46.5191, 390
    ori_x, ori_y, ori_z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(ori_lat, ori_lon, ori_alt)
    print('Origin XYZ: {}, {}, {}'.format(ori_x, ori_y, ori_z))
    origin = np.array([ori_x, ori_y, ori_z], dtype=np.float64)
    return origin


def apply_offset(pc: np.ndarray, offset: np.ndarray, scale: float = 1.0, nodata_value: float = None):
    """
    Apply shift to the point cloud data.
    @param pc:              [N, X] point cloud. (X >= 3, the fourth and later columns are non-coordinates)
    @param offset:          [3] offset vector.
    @param scale:           Float number for scaling.
    @param nodata_value:    Float number for nodata value.
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


def pc_local_search(big_pc: torch.Tensor, ref_patch: torch.Tensor, nodata_value: float = -1.0):
    """
    Point cloud local search based on XYZ boundaries.
    @param big_pc:          [N, D] point cloud. (D >= 3, the fourth and later columns are non-coordinates)
    @param ref_patch:       [K, 3] point cloud as reference.
    @param nodata_value:    Float number for nodata value.
    """
    if ref_patch.numel() == torch.sum(ref_patch == nodata_value).item():
        selected_pc = []
    else:
        xyz_max, _ = ref_patch[ref_patch[:, 1] != nodata_value].max(dim=0)  # [3]
        xyz_min, _ = ref_patch[ref_patch[:, 1] != nodata_value].min(dim=0)  # [3]

        selected_pc = big_pc[(big_pc[:, 0] <= xyz_max[0]) & (big_pc[:, 1] <= xyz_max[1]) & (big_pc[:, 2] <= xyz_max[2])
                             & (big_pc[:, 0] >= xyz_min[0]) & (big_pc[:, 1] >= xyz_min[1]) & (big_pc[:, 2] >= xyz_min[2])
                             ]  # [M, X]
        if len(selected_pc) == 0:
            selected_pc = []
    return selected_pc


def split_scene_coord(sc: np.ndarray, block_h: int, block_w: int):
    """
    Split the scene coordinate associated with image pixels.
    @param sc: [H, W, 3] scene coordinates
    @param block_h: block size in height direction
    @param block_w: block size in width direction
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


def convert_to_tensor(data: np.ndarray, cuda=False, retain_tensor=False, float16=False):
    """Try making float 16 tensor."""
    if float16:
        data_tensor = torch.tensor(data).half()
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


def sc_query(sc: torch.Tensor, pc: torch.Tensor, nodata_value: float = -1.0):
    """
    Query the scene coord in the given point cloud.
    @param sc: [H, W, 3]
    @param pc: [N, 4]
    @param nodata_value:    Float number for nodata value.
    """
    h, w = sc.shape[:2]

    pc = pc.clone()

    sc = sc.reshape(-1, 3)  # [K, 3]
    mask_nodata = sc[:, 0] == nodata_value

    sc_cdist = sc[torch.logical_not(mask_nodata)]  # [K', 3]
    pc_cdist = pc[:, :3]  # [N, 3]

    qeury2pc_dist = torch.cdist(sc_cdist, pc_cdist, p=2.0)  # [K', N]
    idx_cloest_pt_in_pc = qeury2pc_dist.argmin(dim=1).cpu().numpy()  # [K']

    semantics_label = -torch.ones(h*w).to(sc.device)  # [H*W]
    semantics_label[torch.logical_not(mask_nodata)] = pc[idx_cloest_pt_in_pc, -1].float()

    semantics_label = semantics_label.reshape(h, w).cpu().numpy()

    return semantics_label


def main():
    downsample_rate = 1
    scale = 1.0
    block_h, block_w = 48, 48  # GPU memory hungry
    origin = get_ecef_origin()

    # read point cloud with semantic label data from .npy file
    _big_pc_label = np.load("big_pc_label.npy")  # [X, 4]
    _big_pc_label = apply_offset(_big_pc_label, origin, scale, nodata_value=None)  # [X, 4]

    flag_tensor, _big_pc_label_tensor = convert_to_tensor(_big_pc_label, cuda=True, retain_tensor=True)
    assert flag_tensor, "Cannot build tensor for the original data (w/ offset)!"

    file_ls = os.listdir('scene_coord')
    file_ls = ['_'.join(item.split('_')[:-1]) for item in file_ls]
    file_ls = np.unique(file_ls).tolist()
    print(file_ls)

    for idx_dp, file_name in tqdm(enumerate(file_ls)):

        time_start = time.time()
        # load ray-traced point cloud and convert ECEF wgs84 into LV95
        _sc = np.load('scene_coord/{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
        raw_image = io.imread('scene_coord/{:s}_img.png'.format(file_name))[:, :, :3]
        _sc = _sc[::downsample_rate, ::downsample_rate, :]  # [H, W, 3]
        _sc_raw_size = _sc.shape  # [H, W, 3]
        _sc = apply_offset(_sc.reshape(-1, 3), origin, scale, nodata_value=-1).reshape(_sc_raw_size)  # [H, W, 3], numpy array

        flag_tensor, sc = convert_to_tensor(_sc, cuda=True, retain_tensor=True)  # bool, [K, 3] tensor
        assert flag_tensor, "Scene coordinates at {:s} cannot be converted into tensor!".format(file_name)

        _sc_split = split_scene_coord(_sc, block_h, block_w)  # [rows, cols, b_h, b_w, 3]

        flag_tensor, _sc_split = convert_to_tensor(_sc_split, cuda=True, retain_tensor=True)
        assert flag_tensor

        semantics_label_ls = [[[] for _ in range(_sc_split.shape[1])] for _ in range(_sc_split.shape[0])]
        for row in range(_sc_split.shape[0]):
            for col in range(_sc_split.shape[1]):
                selected_pc = pc_local_search(_big_pc_label_tensor, _sc_split[row, col].reshape(-1, 3),
                                              nodata_value=-1)  # [X, 4]
                if len(selected_pc):
                    semantic_label = sc_query(_sc_split[row, col], selected_pc, nodata_value=-1.0)
                else:
                    semantic_label = -np.ones_like(_sc_split[row, col].cpu().numpy())[:, :, 0]
                semantics_label_ls[row][col] = semantic_label

        semantics_label = np.block(semantics_label_ls)
        np.save('semantics_label_{:s}'.format(file_name), semantics_label)

        tiem_elapsed = time.time() - time_start
        print("Semantics label recovery time: {:.1f}s, unique labels: {}".format(tiem_elapsed, np.unique(semantics_label)))

        fig, axes = plt.subplots(1, 2)
        axes[0].axis('off')
        axes[0].imshow(raw_image)

        axes[1].axis('off')
        axes[1].imshow(semantics_label)
        plt.savefig('semantics_label_{:s}.png'.format(file_name), bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    main()
