import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from open3d.cpu.pybind.utility import Vector3dVector
from semantics_recovery_adaptive_super_speed import *
from tqdm import tqdm
import time


# initialization
_big_pc_label = np.load("big_pc_label.npy")  # [X, 4]
bound_xyz_min = _big_pc_label[:, :3].min(axis=0)  # [3]
bound_xyz_max = _big_pc_label[:, :3].max(axis=0)  # [3]
offset_center = (bound_xyz_max + bound_xyz_min) / 2  # [3]
interval_xyz = bound_xyz_max - bound_xyz_min  # [3]
scale = 1.0
nodata_value = -1.0
file_ls = all_path('scene_coord', filter_list=['.png'])
output_path = os.pardir + '/semantics-recovery/scene_coord/processing_data'

for idx_dp, file_name in enumerate(file_ls):
    time_start = time.time()
    # load npy data
    _sc = np.load('{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
    label = np.load('scene_coord/semantics_label_{:s}.npy'.format(file_name.split('/')[-1]))  # [480, 720, 3]
    _sc_raw_size = _sc.shape  # [H, W, 3]
    _sc = apply_offset(_sc.reshape(-1, 3), offset_center, scale, nodata_value=-1).reshape(
        _sc_raw_size)  # [H, W, 3], numpy array
    _sc_pts = _sc.reshape(-1, 3)

    # remove nodata point
    flag_nodata = _sc_pts[:, 1] != nodata_value  # [3]
    nbr_pts_nodata = sum(flag_nodata != True)
    selected_pts = _sc_pts[flag_nodata]

    # create open3D geometry entity
    selected_pts = Vector3dVector(selected_pts)
    sc_pc = o3d.geometry.PointCloud(points=selected_pts)
    nbr_pts_init = len(sc_pc.points)

    # clean the data by remove statistical outlier
    sc_cleaned, idx = sc_pc.remove_statistical_outlier(100, 10)
    nbr_pts_remain = len(sc_cleaned.points)
    nbr_pts_removed = nbr_pts_init - nbr_pts_remain

    # recover bool flag for removed data in (480, 720) format
    # needs to be improved

    # flag_in_PC = [False] * (480 * 720)
    # for i in idx:
    #     flag_in_PC[i] = True
    # flag_removed = [False if flag_nodata[i] == False else flag_in_PC.pop(0) for i in range(480 * 720)]
    # flag_removed = np.array(flag_removed).reshape(480, 720)

    # calculate the coordinate of the bounding box of the image
    axis_aligned_bounding_box = sc_cleaned.get_axis_aligned_bounding_box()
    box_points = axis_aligned_bounding_box.get_box_points()
    box_points = np.asarray(box_points)
    print('axis_aligned_bounding_box: {}'.format(box_points))

    # save processing data
    np.savez(output_path + '/{:s}_output_info.npz'.format(file_name.split('/')[-1]),
             nbr_pts_nodata=nbr_pts_nodata, nbr_pts_removed=nbr_pts_removed,
             box_points=box_points)


    # np.savez(output_path + '/{:s}_output_info.npz'.format(file_name.split('/')[-1]),
    #          nbr_pts_nodata=nbr_pts_nodata, nbr_pts_removed=nbr_pts_removed,
    #          box_points=box_points, flag_removed = flag_removed)

    time_elapsed = time.time() - time_start
    print("Semantics label recovery time: {:.1f}s".format(time_elapsed))
