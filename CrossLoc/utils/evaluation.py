import logging
import os
import pdb

import cv2

import torch
import torch.nn.functional as F
from torch import multiprocessing as mp

import dsacstar
import numpy as np
from typing import Tuple
from dataloader.dataloader import CamLocDataset
from networks.networks import TransPoseNet, Network
from utils.learning import pick_valid_points


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
        else:
            raise NotImplementedError

    flag_coord = task == 'coord'
    flag_depth = task == 'depth'
    flag_normal = task == 'normal'
    flag_semantics = task == 'semantics'

    batch_size = 1 if flag_coord else 4
    eval_set = CamLocDataset(data_to_load, coord=flag_coord, depth=flag_depth, normal=flag_normal,
                             semantics=flag_semantics, mute=mute,
                             augment=False, grayscale=grayscale, raw_image=True, fullsize=fullsize)
    eval_set_loader = torch.utils.data.DataLoader(eval_set, batch_size=batch_size, shuffle=False,
                                                  num_workers=min(mp.cpu_count() // 2, 6),
                                                  pin_memory=True)
    logging.info("This evaluation dataloader has {:d} data points in total.".format(len(eval_set)))

    return eval_set, eval_set_loader


def config_network(scene, task, tiny, grayscale, uncertainty, fullsize, network_in, num_enc=0):
    """
    Configure evaluation network.
    """
    if 'urbanscape' in scene.lower() or 'naturescape' in scene.lower():
        if task == 'coord':
            num_task_channel = 3
        elif task == 'normal':
            num_task_channel = 2
        elif task == 'depth':
            num_task_channel = 1
        elif task == 'semantics':
            num_task_channel = 6
        else:
            raise NotImplementedError
        if uncertainty is None:
            num_pos_channel = 0
        elif uncertainty == 'MLE':
            num_pos_channel = 1
        else:
            raise NotImplementedError
        if task == 'semantics' and uncertainty is not None:
            raise NotImplementedError
        if task == 'semantics' and not fullsize:
            raise NotImplementedError
        network = TransPoseNet(torch.zeros(num_task_channel), tiny, grayscale, num_task_channel=num_task_channel,
                               num_pos_channel=num_pos_channel,
                               enc_add_res_block=2, dec_add_res_block=2, full_size_output=fullsize,
                               num_mlr=num_enc)
    else:
        network = Network(torch.zeros(3), tiny)

    network.load_state_dict(torch.load(network_in), strict=True)
    logging.info("Successfully loaded %s." % network_in)
    network = network.cuda()
    network.eval()

    return network


def get_pose_err(gt_pose: np.ndarray, est_pose: np.ndarray) -> Tuple[float, float]:
    """
    Compute translation and rotation error between two 4x4 transformation matrices.
    """
    transl_err = np.linalg.norm(gt_pose[0:3, 3] - est_pose[0:3, 3])

    rot_err = est_pose[0:3, 0:3].T.dot(gt_pose[0:3, 0:3])
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1, 3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis=1), -1) / np.pi * 180.
    rot_err = rot_err[0]
    return transl_err, rot_err


def scene_coords_eval(scene_coords, gt_coords, gt_pose, nodata_value, focal_length, image_h, image_w,
                      hypotheses, threshold, inlier_alpha, max_pixel_error, output_subsample) \
        -> Tuple[float, float, list, list]:
    """
    Evaluate predicted scene coordinates. Batch size must be one.
    DSAC* PnP solver is adopted. Code reference: https://github.com/vislearn/dsacstar.
    @param scene_coords             [1, 3, H, W], predicted scene coordinates.
    @param gt_coords                [1, 3, H, W], ground-truth scene coordinates.
    @param gt_pose                  [1, 4, 4] cam-to-world matrix.
    @param nodata_value             Nodata value.
    @param focal_length             Camera focal length.
    @param image_h                  Image height.
    @param image_w                  Image width.
    @param hypotheses               DSAC* PnP solver parameter.
    @param threshold                DSAC* PnP solver parameter.
    @param inlier_alpha             DSAC* PnP solver parameter.
    @param max_pixel_error          DSAC* PnP solver parameter.
    @param output_subsample         DSAC* PnP solver parameter.

    @return: t_err, r_err, est_xyz, coords_error for has-data pixels
    """
    gt_pose = gt_pose[0].cpu()

    """metrics on camera pose"""
    # compute 6D camera pose
    out_pose = torch.zeros((4, 4))
    scene_coords = scene_coords.cpu()
    dsacstar.forward_rgb(
        scene_coords,
        out_pose,
        hypotheses,
        threshold,
        focal_length,
        float(image_w / 2),  # principal point assumed in image center
        float(image_h / 2),
        inlier_alpha,
        max_pixel_error,
        output_subsample)

    # calculate pose error
    t_err, r_err = get_pose_err(gt_pose.numpy(), out_pose.numpy())

    # estimated XYZ position
    est_xyz = out_pose[0:3, 3].tolist()

    """metrics on regression error"""
    scene_coords = scene_coords.view(scene_coords.size(0), 3, -1)  # [1, 3, H*W]
    gt_coords = gt_coords.view(gt_coords.size(0), 3, -1)  # [1, 3, H*W]
    mask_gt_coords_valdata = pick_valid_points(gt_coords, nodata_value, boolean=True)  # [1, H*W]

    coords_error = torch.norm(gt_coords - scene_coords, dim=1, p=2)  # [1, H*W]
    coords_error_valdata = coords_error[mask_gt_coords_valdata].tolist()  # [X]

    print("\nRotation Error: %.2f deg, Translation Error: %.1f m, Mean coord prediction error: %.1f m" % (
        r_err, t_err, np.mean(coords_error_valdata)))
    return t_err, r_err, est_xyz, coords_error_valdata


class SemanticsEvaluator(object):
    """
    Helper to evaluate semantics segmentation performance.
    Reference: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,)*2)

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                    np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                    np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

    def reset(self):
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def semantic_eval(semantic_logits, gt_label, mute=False):
    """
    Evaluate semantics segmentation result. The size of logits and label are the same as the raw image.
    @param semantic_logits:     [B, 6, H, W]
    @param gt_label:            [B, 1, H, W]
    @param mute:                flag
    @return:
    """

    gt_label = gt_label.squeeze(1)  # [B, H, W]
    class_prediction = torch.argmax(F.log_softmax(semantic_logits, dim=1), dim=1).cpu()  # [B, H, W]
    assert gt_label.shape == class_prediction.shape

    miou_ls, fwiou_ls, acc_ls = [], [], []  # [B]
    evaluator = SemanticsEvaluator(6)
    for this_gt_label, this_class_pred in zip(gt_label.cpu().numpy(), class_prediction.cpu().numpy()):
        evaluator.reset()
        evaluator.add_batch(this_gt_label, this_class_pred)
        miou_ls.append(evaluator.Mean_Intersection_over_Union())
        fwiou_ls.append(evaluator.Frequency_Weighted_Intersection_over_Union())
        acc_ls.append(evaluator.Pixel_Accuracy())
    miou_ls, fwiou_ls, acc_ls = np.array(miou_ls), np.array(fwiou_ls), np.array(acc_ls)
    if not mute:
        print("Metrics within the batch: mean accuracy: {:.2f}%, mean IoU: {:.2f}%, frequency weighted IoU: {:.2f}%".
            format(acc_ls.mean() * 100, miou_ls.mean() * 100, fwiou_ls.mean() * 100))

    return class_prediction, miou_ls, fwiou_ls, acc_ls

