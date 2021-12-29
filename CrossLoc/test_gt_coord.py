import pdb
import glob
import torch
import numpy as np

import argparse
import os
import multiprocessing as mp
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.learning import get_nodata_value, set_random_seed
from dataloader.dataloader import CamLocDataset
from utils.evaluation import scene_coords_eval, semantic_eval

from typing import Tuple, Union


def _config_parser():
    """
    Task specific argument parser
    """
    parser = argparse.ArgumentParser(
        description='Initialize a scene coordinate regression network.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    """General training parameter"""
    # Dataset and dataloader
    parser.add_argument('scene', help='name of a scene in the dataset folder')

    parser.add_argument('--task', type=str,
                        help='specify the single regression task, should be "coord", "depth" or "normal"')

    parser.add_argument('--section', type=str, nargs='+', default=['all'],
                        help='Dataset to test model performance, could be val or test.')

    # Network structure
    parser.add_argument('--fullsize', '-fullsize', action='store_true',
                        help='to output fullsize prediction w/o down-sampling.')

    """DSAC* PnP solver parameters"""
    # Default values are used
    parser.add_argument('--hypotheses', '-hyps', type=int, default=64,
                        help='number of hypotheses, i.e. number of RANSAC iterations')

    parser.add_argument('--threshold', '-t', type=float, default=10,
                        help='inlier threshold in pixels (RGB) or centimeters (RGB-D)')

    parser.add_argument('--inlieralpha', '-ia', type=float, default=100,
                        help='alpha parameter of the soft inlier count; '
                             'controls the softness of the hypotheses score distribution; lower means softer')

    parser.add_argument('--maxpixelerror', '-maxerrr', type=float, default=100,
                        help='maximum reprojection (RGB, in px) or 3D distance (RGB-D, in cm) error when '
                             'checking pose consistency towards all measurements; '
                             'error is clamped to this value for stability')

    opt = parser.parse_args()

    return opt


def config_dataloader(scene, task, grayscale, section_keyword, fullsize, mute=False):
    """
    Configure evaluation dataloader.
    """
    if 'urbanscape' in scene.lower() or 'naturescape' in scene.lower():
        pass
    else:
        raise NotImplementedError

    # fullsize adaptive tweaking
    if task == 'semantics':
        _scene = scene
        assert fullsize
    else:
        _scene = scene + '-fullsize' if fullsize else scene

    data_to_load = "./datasets/" + scene + "/" + section_keyword

    if os.path.exists(data_to_load):
        if mute:
            pass
        else:
            print("Loading evaluation data at {:s}".format(data_to_load))
    else:
        print("Loading special section {:s}".format(section_keyword))
        if section_keyword == 'test_real_all':
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_real",
                            "./datasets/" + scene + "/" + "test_drone_real"]
        elif section_keyword == "real_all":
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_real",
                            "./datasets/" + scene + "/" + "test_drone_real",
                            "./datasets/" + scene + "/" + "train_drone_real"]
        elif section_keyword == "test_sim_all":
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_sim",
                            "./datasets/" + scene + "/" + "val_sim",
                            "./datasets/" + scene + "/" + "test_drone_sim"]
        elif section_keyword == "sim_all":
            data_to_load = ["./datasets/" + scene + "/" + "val_drone_sim",
                            "./datasets/" + scene + "/" + "val_sim",
                            "./datasets/" + scene + "/" + "test_drone_sim",
                            "./datasets/" + scene + "/" + "train_sim"]
        elif section_keyword == "all":
            data_to_load = ["./datasets/" + scene + "/" + "train_sim",
                            "./datasets/" + scene + "/" + "val_sim",
                            "./datasets/" + scene + "/" + "train_drone_real",
                            "./datasets/" + scene + "/" + "train_oop_drone_real",
                            "./datasets/" + scene + "/" + "val_drone_real",
                            "./datasets/" + scene + "/" + "val_oop_drone_real",
                            "./datasets/" + scene + "/" + "test_drone_real",
                            "./datasets/" + scene + "/" + "test_oop_drone_real"]
        else:
            raise NotImplementedError

    flag_coord = task == 'coord'
    flag_depth = True
    flag_normal = task == 'normal'
    flag_semantics = task == 'semantics'

    batch_size = 1 if flag_coord else 4
    eval_set = CamLocDataset(data_to_load, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                             semantics=flag_semantics, mute=mute,
                             augment=False, grayscale=grayscale, raw_image=True, fullsize=fullsize)
    eval_set_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=min(mp.cpu_count() // 2, 6),
                                                  pin_memory=True)

    return eval_set, eval_set_loader


def main():
    """
    Main function.
    """

    """Initialization"""
    set_random_seed(2021)
    opt = _config_parser()
    print(opt)

    scene = opt.scene
    task = "coord"
    """Loop over network weights"""
    # initialization
    nodata_value = get_nodata_value(scene)
    if opt.section[0] == 'all' and len(opt.section) == 1:
        opt.section = 'all'

    """Loop over dataset sections"""
    testing_log = os.path.abspath("eval_gt_dataset_{:s}.txt".format(scene))
    print("{:s} Evaluating over dataset {:s} {:s}".format('*'*20, scene, '*'*20))
    eval_set, eval_set_loader = config_dataloader(scene, task, False, opt.section, False, mute=True)

    if task == 'coord':
        t_err_ls, r_err_ls, est_xyz_ls, avg_depth_ls = [], [], [], []
    else:
        raise NotImplementedError
    file_name_ls = []

    for j, (image, gt_pose, gt_label, focal_length, file_name) in enumerate(tqdm(eval_set_loader,
                                                                                 desc='GT evaluation')):
        """Data pre-processing"""
        focal_length = float(focal_length.view(-1)[0])
        """
        @image         [B, C, H, W] ---> [B, 3, 480, 720] by default w/o augmentation, RGB image
        @gt_pose       [B, 4, 4], camera to world matrix
        @gt_label      [B, C, H_ds, W_ds] ---> [B, C, 60, 90] by default w/o augmentation
        @focal_length  [1], adapted to augmentation
        @file_name     a list size of B
        """
        # cam_mat = get_cam_mat(image.size(3), image.size(2), focal_length)
        # gt_pose = gt_pose.cuda()
        # gt_label = gt_label.cuda()
        file_name = os.path.basename(file_name[0])
        file_name_ls.append(file_name)

        with torch.no_grad():
            """Forward pass"""
            if isinstance(gt_label, dict):
                predictions = gt_label['coord']  # [1, 3, H, W]
                depth = gt_label['depth']  # [1, 1, H, W]
                gt_label = predictions  # for compatibility
            else:
                predictions = gt_label  # [1, 3, H, W]

            """Metrics evaluation"""
            if task == 'coord':
                # predictions = gt_label  # debug only!
                t_err, r_err, est_xyz, _ = scene_coords_eval(
                    predictions, gt_label, gt_pose, nodata_value, focal_length,
                    image.size(2), image.size(3), opt.hypotheses, opt.threshold,
                    opt.inlieralpha, opt.maxpixelerror, 8)
                t_err_ls.append(t_err)
                r_err_ls.append(r_err)
                est_xyz_ls.append(est_xyz)

                avg_depth = depth[depth != nodata_value].mean().item()
                avg_depth_ls.append(avg_depth)
            else:
                raise NotImplementedError

    print("{:s} Evaluating over scene {:s} is done!{:s}".format('*'*20, scene, '*'*20))

    if task == "coord":
        t_err_ls = np.array(t_err_ls)  # [N]
        r_err_ls = np.array(r_err_ls)  # [N]
        file_name_ls = np.array(file_name_ls)  # [N]
        est_xyz_ls = np.stack(est_xyz_ls, axis=0)  # [N, 3]
        avg_depth_ls = np.stack(avg_depth_ls)  # [N]

        eval_str = '\nAccuracy:'
        eval_str += "\nMin Error: %.1f deg, %.2f m" % (np.min(r_err_ls), np.min(t_err_ls))
        eval_str += "\nMax Error: %.1f deg, %.2f m" % (np.max(r_err_ls), np.max(t_err_ls))
        eval_str += "\nMedian Error: %.1f deg, %.2f m" % (np.median(r_err_ls), np.median(t_err_ls))
        eval_str += "\nMean Errors: %.1f plus-minus %.1f deg, %.2f plus-minus %.2f m" % (
            np.mean(r_err_ls), np.std(r_err_ls), np.mean(t_err_ls), np.std(t_err_ls))
        eval_str += "\nMean Depth: %.2f plus-minus %.2fm, Median Depth: %.2fm" % (
            np.mean(avg_depth_ls), np.std(avg_depth_ls), np.median(avg_depth_ls))

        print(eval_str)

        with open(testing_log, 'a') as f:
            f.write("{:s} Evaluation on scene {:s} {:s}".format('=' * 20, scene, '=' * 20) + '\n')
            f.write(eval_str)
            f.write('\n')

        eval_results = np.stack([t_err_ls, r_err_ls, avg_depth_ls], axis=1)  # [N, 3]
        np.savez(os.path.abspath("eval_gt_dataset_{:s}.npz".format(scene)), results=eval_results,
                 file_name=file_name_ls)

    print("GT-data evaluation finished. Please find the log at {:s}".format(testing_log))


if __name__ == "__main__":
    main()
