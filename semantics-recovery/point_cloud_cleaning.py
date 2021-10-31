import open3d as o3d
from open3d.cpu.pybind.utility import Vector3dVector
from semantics_recovery_adaptive_super_speed import *


def mkdir(output_path, folder_ls):
    for folder in folder_ls:
        output_folder = os.path.exists(output_path + '/' + folder)
        if not output_folder:
            os.makedirs(output_path + '/' + folder)


def main():
    # initialization

    nodata_value = -1.0
    input_path = '/media/shanci/Samsung_T5/TOPO_datasets/EPFL'
    output_path = '/media/shanci/Samsung_T5/processing_data/EPFL_no_data_cleaning'
    file_ls, folder_ls = all_path(input_path, filter_list=['.npy'])

    # create output folder structure
    input_path_len = len(input_path.split('/'))
    folder_ls = ['/'.join(folder.split('/')[input_path_len:]) for folder in folder_ls]
    folder_ls = np.unique(folder_ls).tolist()
    mkdir(output_path, folder_ls)

    for idx_dp, file_name in tqdm(enumerate(file_ls[:2000]), desc='data cleaning'):
        # time_start = time.time()
        # load npy data
        _sc = np.load('{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
        # label = np.load(input_path + '/semantics_label_{:s}.npy'.format(file_name.split('/')[-1]))  # [480, 720, 3]
        _sc_raw_size = _sc.shape  # [H, W, 3]
        _sc_pts = _sc.reshape(-1, 3)

        # remove nodata point
        flag_nodata = _sc_pts[:, 1] != nodata_value  # [H*W]
        nbr_pts_nodata = sum(flag_nodata != True)
        selected_pts = _sc_pts[flag_nodata]

        # create open3D geometry entity
        selected_pts = Vector3dVector(selected_pts)
        sc_pc = o3d.geometry.PointCloud(points=selected_pts)
        nbr_pts_init = len(sc_pc.points)

        # # clean the data by remove statistical outlier
        # sc_cleaned, idx = sc_pc.remove_statistical_outlier(100, 10)
        # nbr_pts_remain = len(sc_cleaned.points)
        # nbr_pts_removed = nbr_pts_init - nbr_pts_remain

        # # recover bool flag for removed data in (480, 720) format
        # flag_removed = np.zeros_like(flag_nodata)
        # valdata_slice = flag_removed[flag_nodata]
        # valdata_slice[idx] = True
        # flag_removed[flag_nodata] = valdata_slice
        # flag_removed = flag_removed.reshape(480, 720)

        # calculate the coordinate of the bounding box of the image
        # axis_aligned_bounding_box = sc_cleaned.get_axis_aligned_bounding_box()
        axis_aligned_bounding_box = sc_pc.get_axis_aligned_bounding_box()
        box_points = axis_aligned_bounding_box.get_box_points()
        box_points = np.asarray(box_points)
        # print('axis_aligned_bounding_box: {}'.format(box_points))

        # save processing data
        path = output_path + '/' +'{:s}_output_info.npz'.format('/'.join(file_name.split('/')[input_path_len:]))
        np.savez(path, nbr_pts_nodata=nbr_pts_nodata, box_points=box_points)
        # np.savez(path, nbr_pts_nodata=nbr_pts_nodata, nbr_pts_removed=nbr_pts_removed,
        #          box_points=box_points, flag_removed=flag_removed)

        # time_elapsed = time.time() - time_start
        # print("block location time: {:.1f}s".format(time_elapsed))


if __name__ == '__main__':
    main()
