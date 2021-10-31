import os

import laspy
import pyproj
import numpy as np

from tqdm import tqdm
from reframeTransform import ReframeTransform


def geographic_to_ecef(lon, lat, alt, transformer):
    # Careful: here we need to use lat,lon
    x, y, z = transformer.transform(lat, lon, alt)
    return [x, y, z]


def coordinate_transform(big_pic, transformer):
    # load ray-traced point cloud and convert
    r = ReframeTransform()
    big_pic_wgs84 = np.empty_like(big_pic)
    for idx, pt in tqdm(enumerate(big_pic), desc="Translating coordiante system"):
        if pt[0] == -1:
            pt_wgs84 = pt
        else:
            # import pdb
            # print('Original:', pt)
            # pt_wgs84 = r.transform(pt.copy(), 'lv95', 'ln02', 'wgs84', 'ln02')
            # pt_wgs84 = r.transform(pt_wgs84, 'wgs84', 'ln02', 'wgs84', 'wgs84')
            # pt_wgs84 = geographic_to_ecef(*pt_wgs84)
            # print("Pipeline1:", pt_wgs84)

            """ this should work """
            pt_wgs84 = r.transform(pt.copy(), 'lv95', 'ln02', 'wgs84', 'wgs84')
            pt_wgs84 = geographic_to_ecef(*pt_wgs84, transformer)
            # print("Pipeline2:", pt_wgs84)

        big_pic_wgs84[idx] = pt_wgs84
    return big_pic_wgs84


def main():

    # initialize transformer for coordinate transform geographic_to_ecef
    transformer = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978")

    # load raw las and turn into 3D point array
    _big_pc = []

    las_ls = os.listdir('EPFL-surface3d/swiss')
    for idx, las_name in enumerate(las_ls):
        las = laspy.read(os.path.join('EPFL-surface3d/swiss', las_name))
        las = np.stack([las.x, las.y, las.z, np.array(las.classification)], axis=1)
        _big_pc.extend(las)

    _big_pc = np.array(_big_pc) # [N, 4]
    _big_pc = np.ascontiguousarray(_big_pc)  # [N, 4], columns: lv95-East, lv95-North, ln02-Height, label
    big_pc = _big_pc[:, :3]  # [N, 3]
    label = _big_pc[:, 3].reshape(-1, 1) #[N,1]
    # from swisstopo coordinate system to wgs84
    big_pc_ecef = coordinate_transform(big_pc, transformer)
    big_pc_label_ecef = np.concatenate((big_pc_ecef, label), axis=1)
    # save big_pc_label.npy file
    np.save('big_pc_label.npy', big_pc_label_ecef)


if __name__ == '__main__':
	main()
