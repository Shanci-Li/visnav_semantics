import numpy as np
import pyproj
from semantics_recovery_adaptive_super_speed import *
from reframeTransform import ReframeTransform


def ecef_to_geographic(x, y, z):
    # Careful: here we need to use lat,lon
    lat, lon, alt = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979").transform(x, y, z)
    return [lon, lat, alt]


def main():
    # initialization
    input_path = '/media/shanci/Samsung_T5/processing_data/comballaz'
    file_ls, folder_ls = all_path(input_path, filter_list=['.npz'])
    nbr_pts_nodata = 0
    nbr_pts_removed = 0
    r = ReframeTransform()
    bound_wgs84_max = []
    bound_wgs84_min = []

    _big_pc_label = np.load("big_pc_label.npy")  # [X, 4]
    bound_xyz_min = _big_pc_label[:, :3].min(axis=0)  # [3]
    bound_xyz_max = _big_pc_label[:, :3].max(axis=0)  # [3]
    offset_center = (bound_xyz_max + bound_xyz_min) / 2  # [3]

    time_start = time.time()

    for idx_dp, file_name in tqdm(enumerate(file_ls), desc='data integrating'):
        # load npy data
        info_data = np.load('{:s}_info.npz'.format(file_name))  # [480, 720, 3]
        nbr_pts_nodata += info_data['nbr_pts_nodata']
        nbr_pts_removed += info_data['nbr_pts_removed']
        box_pts = info_data['box_points']
        box_pts_ecef = apply_offset(box_pts, -1 * offset_center, scale=1.0, nodata_value=None)
        box_pts_wgs84 = np.empty_like(box_pts_ecef)

        for idx, pt in enumerate(box_pts_ecef):
            # Careful: Use lon,lat in reframeTransform
            lon, lat, alt = ecef_to_geographic(*pt)
            # wgs84_coord = r.transform([lon, lat, alt], 'wgs84', 'wgs84', 'wgs84', 'ln02')
            wgs84_coord = [lon, lat, alt]
            box_pts_wgs84[idx] = wgs84_coord
        xy_max = box_pts_wgs84[:, 0:2].max(axis=0)
        xy_min = box_pts_wgs84[:, 0:2].min(axis=0)
        bound_wgs84_max.append(xy_max)
        bound_wgs84_min.append(xy_min)

    bound_wgs84_max = np.array(bound_wgs84_max).max(axis=0)
    bound_wgs84_min = np.array(bound_wgs84_min).min(axis=0)
    print('boundary corner coordinate: \n'
          'bound_wgs84_max: {}\n'
          'bound_wgs84_min: {}'.format(bound_wgs84_max, bound_wgs84_min))

    np.savez('data/comballaz_block_location_info.npz', bound_wgs84_max=bound_wgs84_max, bound_wgs84_min=bound_wgs84_min)

    time_elapsed = time.time() - time_start
    print("block location time: {:.1f}s".format(time_elapsed))


if __name__ == '__main__':
    main()
