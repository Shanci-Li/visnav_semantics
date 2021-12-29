import pdb
import glob
import torch
import numpy as np

import argparse
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.learning import get_nodata_value, set_random_seed
from utils.evaluation import config_dataloader, config_network, scene_coords_eval, semantic_eval

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

    parser.add_argument('--grayscale', '-grayscale', action='store_true',
                        help='use grayscale image as model input')

    parser.add_argument('--task', type=str,
                        help='specify the single regression task, should be "coord", "depth" or "normal"')

    parser.add_argument('--section', type=str, nargs='+', default=['val_drone_sim', 'val_drone_real'],
                        help='Dataset to test model performance, could be val or test.')

    # Network structure
    parser.add_argument('--network_in', type=str, default=None,
                        help='file name of a network initialized for the scene')

    parser.add_argument('--tiny', '-tiny', action='store_true',
                        help='Load a model with massively reduced capacity for a low memory footprint.')

    parser.add_argument('--fullsize', '-fullsize', action='store_true',
                        help='to output fullsize prediction w/o down-sampling.')

    """I/O parameters"""
    parser.add_argument('--session', '-sid', default='',
                        help='custom session name appended to output files'
                             'useful to separate different runs of a script')

    parser.add_argument('--search_dir', action='store_true',
                        help='Search local directory for all models. '
                             'Note: most dataset, dataloader & section parameters would be reset and '
                             'automatically deduced from the folder names!')

    parser.add_argument('--min_ckpt_iter', default=None, type=int,
                        help='Minimum checkpoint model training iteration')

    parser.add_argument('--keywords', default=None, nargs='+',
                        help="Keywords to filter out some network weight paths.")

    parser.add_argument('--plot', action="store_true",
                        help="Plot the qualitative results.")

    parser.add_argument('--save_pred', action="store_true",
                        help="Save predicted results.")

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

    """Uncertainty loss parameter"""
    parser.add_argument('--uncertainty', '-uncertainty', default=None, type=str,
                        help='enable uncertainty learning')

    opt = parser.parse_args()

    # Auto-reset parameters, they will be deduced from the searched folders' names
    if opt.search_dir:
        opt.scene = None
        opt.grayscale = None
        opt.task = None
        opt.section = None
        opt.tiny = None
        opt.fullsize = None
        opt.uncertainty = None

        print("search_dir is ON. Model parameters would be read from the folder name...")

    return opt


def _config_weight_path(network_in: Union[str, list], keywords: Union[None, str, list] = None,
                        search_dir: bool = False, min_ckpt_iter: int = 0) -> list:
    """
    Identify the paths to the model weights.
    @param network_in:      One or multiple parental directories or paths to network model weight.
    @param keywords:        One or multiple (union) keywords to search for.
    @param search_dir:      Search for all model weights found in the single specified directory.
    @return: network_paths  One or multiple paths to the network weights filtered by the keywords.
    """

    if isinstance(network_in, list):
        # A list of model weights or directories containing model weights
        _network_in = sorted([os.path.abspath(item) for item in network_in])
        print("To load {:d} network weights...")
        [print(entry) for entry in _network_in]
    elif isinstance(network_in, str):
        # A path to a model weight of a directory containing model weight
        _network_in = [os.path.abspath(network_in)]
        print("To load network weight from {:s}".format(_network_in[0]))
    else:
        raise NotImplementedError

    if search_dir:
        # Turn the directory into a list of sub-directories
        assert len(_network_in) == 1, "_network_in must have one item in search_dir mode. Now it's: {}".format(
            _network_in)
        assert os.path.isdir(_network_in[0]), "_network_in {:s} is not a directory.".format(_network_in[0])
        src_dir = os.path.abspath(_network_in[0])
        _network_in = [os.path.join(src_dir, sub_dir) for sub_dir in os.listdir(src_dir)]

    """Get the exact model weights path"""
    network_paths = []
    for path in _network_in:
        assert os.path.exists(path), "Network input path {:s} is not found.".format(path)
        if os.path.isdir(path):
            # some directory containing weight
            model_path = os.path.join(path, 'model.net')
            if os.path.exists(model_path):
                network_paths.append(model_path)
            ckpt_models = glob.glob(os.path.join(path, 'ckpt_iter*.net'))
            network_paths += ckpt_models
        elif os.path.isfile(path):
            # some designated model path
            if os.path.basename(path).startswith("model") or "ckpt_" in os.path.basename(path):
                if os.path.basename(path).endswith('.net'):
                    network_paths.append(path)
        else:
            raise NotImplementedError

    """Keyword search"""
    if keywords is None:
        pass
    elif isinstance(keywords, str):
        keywords = [keywords]
    elif isinstance(keywords, list):
        pass
    else:
        raise NotImplementedError

    if keywords is not None:
        network_paths_raw = network_paths.copy()
        network_paths = []
        for path in network_paths_raw:
            flags = np.array([keyword in os.path.dirname(path) for keyword in keywords])
            if flags.any():
                network_paths.append(path)
        network_paths = np.sort(np.unique(network_paths)).tolist()

    _network_paths = network_paths.copy()
    network_paths = []
    if min_ckpt_iter is not None:
        if min_ckpt_iter > 0:
            for path in _network_paths:
                if 'ckpt_iter_' in os.path.basename(path):
                    this_iter = int(os.path.basename(path).split('_')[-1].replace('.net', ''))
                    if this_iter > min_ckpt_iter:
                        network_paths.append(path)
    else:
        network_paths = _network_paths

    network_paths.sort()
    print("With the keywords {:}, and min ckpt iter {},".format(keywords, min_ckpt_iter), end=" ")
    print("the following {:d} network weight paths are retrieved:".format(len(network_paths)))
    for idx, path in enumerate(network_paths):
        print("Network weight #{:d}: {:s}".format(idx, path))

    return network_paths


def read_mata_info(folder_name):
    """Read meta info from the folder name."""

    def get_uncertainty_keyword(dirname):
        if 'no_unc' in dirname:
            return None
        elif 'unc-MLE' in dirname:
            return 'MLE'
        else:
            raise NotImplementedError

    """urbanscape-semantics-no_unc-fullsize-e150-lr0.0002-sim_only-sc1.00"""
    scene = folder_name.split('-')[0]
    grayscale = '-gray' in folder_name
    task = folder_name.split('-')[1]

    if 'decoder' in folder_name:
        # after fine-tuning
        # section = ['test_drone_sim', 'test_drone_real', 'test_oop_drone_sim', 'test_oop_drone_real']
        if '-oop-' in folder_name:
            section = ['test_oop_drone_real']
        elif '-ip-' in folder_name:
            section = ['test_drone_real']
        else:
            raise NotImplementedError
    else:
        # after pretraining
        # section = ['val_sim', 'val_drone_sim', 'val_drone_real', 'val_oop_drone_sim', 'val_oop_drone_real']
        section = ['val_drone_sim', 'val_drone_real']  # fast

    tiny = '-tiny' in folder_name
    fullsize = '-fullsize' in folder_name or '-semantics' in folder_name
    uncertainty = get_uncertainty_keyword(folder_name)
    return scene, grayscale, task, section, tiny, fullsize, uncertainty


def main():
    """
    Main function.
    """

    """Initialization"""
    set_random_seed(2021)
    opt = _config_parser()
    print(opt)

    network_paths = _config_weight_path(opt.network_in, opt.keywords, opt.search_dir, opt.min_ckpt_iter)

    """Loop over network weights"""
    for i, network_path in enumerate(network_paths):
        print("{:s} Evaluating network #{:d} {:s}\nPath: {:s}".format('='*20, i, '='*20, network_path))

        # update evaluation if needed
        if opt.search_dir:
            scene, grayscale, task, section, tiny, fullsize, uncertainty = read_mata_info(os.path.basename(
                os.path.dirname(network_path)))
            print("Scene: {:s}, grayscale: {}, task: {:s}, #section: {:d}, tiny: {}, fullsize: {}, uncertainty: {}".
                  format(scene, grayscale, task, len(section), tiny, fullsize, uncertainty))
        else:
            scene, grayscale, task, section, tiny, fullsize, uncertainty = opt.scene, opt.grayscale, opt.task, \
                                                              opt.section, opt.tiny, opt.fullsize, opt.uncertainty

        # initialization
        nodata_value = get_nodata_value(scene)

        if 'decoder_coord_ENCDEC_free_depth_normal_semantics' in network_path or 'crossloc_se' in network_path.lower():
            num_enc = 4
        elif 'decoder_coord_ENCDEC_free_semantics' in network_path:
            num_enc = 2
        elif 'decoder_coord_ENCDEC_free_depth_normal' in network_path or 'crossloc' in network_path.lower():
            num_enc = 3
        else:
            num_enc = 0
        network = config_network(scene, task, tiny, grayscale, uncertainty, fullsize, network_path, num_enc=num_enc)

        testing_log = os.path.join(os.path.dirname(network_path), 'results_{:s}_task_{:s}.txt'.format(
            os.path.basename(network_path), task))

        """Loop over dataset sections"""
        for this_section in section:
            print("{:s} Evaluating over section {:s} {:s}".format('*'*20, this_section, '*'*20))
            eval_set, eval_set_loader = config_dataloader(scene, task, grayscale, this_section, fullsize, mute=True)

            if opt.save_pred:
                pred_dir = os.path.abspath(os.path.join(network_path, '../{:s}_pred_{:s}_{:s}'.format(
                    task, os.path.basename(network_path), this_section)))
                os.makedirs(pred_dir, exist_ok=True)

            if task == 'coord':
                t_err_ls, r_err_ls, est_xyz_ls, coords_error_ls = [], [], [], []
            elif task == 'depth':
                raise NotImplementedError
            elif task == 'normal':
                raise NotImplementedError
            elif task == 'semantics':
                mean_iou_ls = []
                fw_iou_ls = []
                accuracy_ls = []
            else:
                raise NotImplementedError
            file_name_ls = []

            for j, (image, gt_pose, gt_label, focal_length, file_name) in enumerate(tqdm(eval_set_loader,
                                                                                         desc='Network #%d' % i)):
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
                    predictions = network(image.cuda())  # [1, C, H, W]
                    if fullsize:
                        assert predictions.size(2) == image.size(2) and predictions.size(3) == image.size(3)
                        assert predictions.size(2) == gt_label.size(2) and predictions.size(3) == gt_label.size(3)
                    if uncertainty is None:
                        uncertainty_map = None
                    elif uncertainty == 'MLE':
                        predictions, uncertainty_map = torch.split(predictions,
                                                                   [network.num_task_channel, network.num_pos_channel],
                                                                   dim=1)  # typically [1, C, H, W] + [1, 1, H, W]
                    else:
                        raise NotImplementedError
                    if opt.save_pred:
                        torch.save(predictions.squeeze(0),
                                   os.path.join(pred_dir, file_name.replace('.png', '.dat')))

                    """Metrics evaluation"""
                    if task == 'coord':
                        # predictions = gt_label  # debug only!
                        t_err, r_err, est_xyz, coords_error = scene_coords_eval(
                            predictions, gt_label, gt_pose, nodata_value, focal_length,
                            image.size(2), image.size(3), opt.hypotheses, opt.threshold,
                            opt.inlieralpha, opt.maxpixelerror, network.OUTPUT_SUBSAMPLE)
                        t_err_ls.append(t_err)
                        r_err_ls.append(r_err)
                        est_xyz_ls.append(est_xyz)
                        coords_error_ls.append(coords_error)
                    elif task == 'depth':
                        depth_eval()
                    elif task == 'normal':
                        normal_eval()
                    elif task == 'semantics':
                        class_prediction, miou, fwiou, acc = semantic_eval(predictions, gt_label, mute=True)
                        mean_iou_ls.append(miou)
                        fw_iou_ls.append(fwiou)
                        accuracy_ls.append(acc)

                        """plot the batch results"""
                        if opt.plot:
                            fig, axes = plt.subplots(len(acc), 3, figsize=(3, len(acc)))
                            for row in range(len(acc)):

                                axes[row, 0].axis('off')
                                axes[row, 0].imshow(image[row].numpy().transpose(1, 2, 0))

                                axes[row, 1].axis('off')
                                axes[row, 1].imshow(class_prediction[row], vmin=0, vmax=6)

                                axes[row, 2].axis('off')
                                axes[row, 2].imshow(gt_label[row][0], vmin=0, vmax=6)

                            plt.subplots_adjust(wspace=0.01, hspace=0.01)
                            plt.savefig(os.path.abspath(os.path.join(
                                network_path, '..', 'sm_section_{:s}_batch_{:d}'.format(this_section, j))),
                                bbox_inches='tight', pad_inches=0.1, dpi=300)
                            plt.close(fig)

                            if j > 10:
                                break
                    else:
                        raise NotImplementedError
                    # pdb.set_trace()

                    """Save to csv"""

            print("{:s} Evaluating over section {:s} is done!{:s}".format('*'*20, this_section, '*'*20))

            if task == "coord":
                t_err_ls = np.array(t_err_ls)  # [N]
                r_err_ls = np.array(r_err_ls)  # [N]
                est_xyz_ls = np.stack(est_xyz_ls, axis=0)  # [N, 3]
                coords_error_ls = np.concatenate(coords_error_ls)  # [X]

                pct30_10 = np.sum(np.logical_and(t_err_ls < 30.0, r_err_ls < 10.0))
                pct20_10 = np.sum(np.logical_and(t_err_ls < 20.0, r_err_ls < 10.0))
                pct10_10 = np.sum(np.logical_and(t_err_ls < 10.0, r_err_ls < 10.0))
                pct10_7 = np.sum(np.logical_and(t_err_ls < 10.0, r_err_ls < 7.0))
                pct5 = np.sum(np.logical_and(t_err_ls < 5.0, r_err_ls < 5.0))
                pct3 = np.sum(np.logical_and(t_err_ls < 3.0, r_err_ls < 3.0))

                eval_str = '\nAccuracy:'
                eval_str += '\n30m10deg: %.1f%%\n20m10deg: %.1f%%' % (pct30_10 / len(t_err_ls) * 100,
                                                                      pct20_10 / len(t_err_ls) * 100)
                eval_str += '\n10m7deg: %.1f%%' % (pct10_7 / len(t_err_ls) * 100)
                eval_str += '\n10m10deg: %.1f%%' % (pct10_10 / len(t_err_ls) * 100) + '\n5m5deg: %.1f%%' % (
                            pct5 / len(t_err_ls) * 100)
                eval_str += '\n3m3deg: %.1f%%' % (pct3 / len(t_err_ls) * 100)
                eval_str += "\nMedian Error: %.1f deg, %.2f m" % (np.median(r_err_ls), np.median(t_err_ls))
                eval_str += "\nMean Errors: %.1f plus-minus %.1f deg, %.2f plus-minus %.2f m" % (
                    np.mean(r_err_ls), np.std(r_err_ls), np.mean(t_err_ls), np.std(t_err_ls))
                eval_str += "\nCoordinate regression error: mean {:.1f}, std {:.1f}, median {:.1f}".format(
                    np.mean(coords_error_ls), np.std(coords_error_ls), np.median(coords_error_ls))
                print(eval_str)

                with open(testing_log, 'a') as f:
                    f.write("{:s} Evaluation on section {:s} {:s}".format('=' * 20, this_section, '=' * 20) + '\n')
                    f.write(eval_str)
                    f.write('\n')

                np.save(os.path.join(os.path.dirname(network_path), '{:s}_{:s}_out_xyz_poses.npy'.format(
                    this_section, os.path.basename(network_path))), est_xyz_ls)

                with open(os.path.join(os.path.dirname(network_path), '{:s}_{:s}_out_xyz_poses.txt'.format(
                        this_section, os.path.basename(network_path))), 'w') as f:
                    for file, pose_xyz in zip(file_name_ls, est_xyz_ls):
                        f.write(file + ' {:.2f} {:.2f} {:.2f}'.format(*pose_xyz) + '\n')

            elif task == 'semantics':
                accuracy_ls = np.concatenate(accuracy_ls)
                mean_iou_ls = np.concatenate(mean_iou_ls)
                fw_iou_ls = np.concatenate(fw_iou_ls)

                accuracy_str = "Pixel accuracy, mean: {:.2f}, median: {:.2f}".format(
                    np.mean(accuracy_ls) * 100, np.median(accuracy_ls) * 100)
                print(accuracy_str)

                mean_iou_str = "Mean IoU, mean: {:.2f}, median: {:.2f}".format(
                    np.mean(mean_iou_ls) * 100, np.median(mean_iou_ls) * 100)
                print(mean_iou_str)

                fw_iou_str = "Frequency weighted IoU, mean: {:.2f}, median: {:.2f}".format(
                    np.mean(fw_iou_ls) * 100, np.median(fw_iou_ls) * 100)
                print(fw_iou_str)

                # out_path = os.path.join(os.path.dirname(network_path), 'results_{:s}_{:s}_section_{:s}.txt'.format(
                #     os.path.basename(network_path), task, this_section))
                # with open(out_path, 'w') as f:
                #     f.write(accuracy_str + '\n')
                #     f.write(mean_iou_str + '\n')

                with open(testing_log, 'a') as f:
                    f.write("{:s} Evaluation on section {:s} {:s}".format('=' * 20, this_section, '=' * 20) + '\n')
                    f.write(accuracy_str + '\n')
                    f.write(mean_iou_str + '\n')
                    f.write(fw_iou_str + '\n')
                    f.write('\n')

        print("Network testing finished. Please find the log at {:s}".format(testing_log))


if __name__ == "__main__":
    main()
