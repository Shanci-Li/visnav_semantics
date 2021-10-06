import laspy
import pyproj
import numpy as np

from tqdm import tqdm
from reframeTransform import ReframeTransform


def geographic_to_ecef(lon, lat, alt):
    # Careful: here we need to use lat,lon
    x, y, z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(lat, lon, alt)
    return [x, y, z]

def coordinate_transform(big_pic):
    # load ray-traced point cloud and convert
    r = ReframeTransform()
    big_pic_wgs84 = np.empty_like(big_pic)
    for idx, pt in tqdm(enumerate(big_pic), desc="Translating coordiante system"):
        if pt[0] == -1:
            pt_wgs84 = pt
        else:
            pt_wgs84 = r.transform(pt, 'lv95', 'ln02', 'wgs84', 'ln02')
            pt_wgs84 = r.transform(pt_wgs84, 'wgs84', 'ln02', 'wgs84', 'wgs84')
            pt_wgs84 = geographic_to_ecef(*pt_wgs84)
        big_pic_wgs84[idx] = pt_wgs84
    return big_pic_wgs84



def main():

    # load raw las and turn into 3D point array
    las_1 = laspy.read('2533_1152.las')
    las_2 = laspy.read('2532_1152.las')
    las_1 = np.stack([las_1.x, las_1.y, las_1.z, np.array(las_1.classification)], axis=0)  # [4, N]
    las_2 = np.stack([las_2.x, las_2.y, las_2.z, np.array(las_2.classification)], axis=0)
    _big_pc = np.concatenate((las_1, las_2), axis=1)  # [4, 2N]
    _big_pc = np.ascontiguousarray(_big_pc.transpose())  # [2N, 4], columns: lv95-East, lv95-North, ln02-Height, label
    big_pc = _big_pc[:, :3]  # [2N, 3]
    label = _big_pc[:, 3].reshape(-1,1) #[2N,1]
    # from swisstopo coordinate system to wgs84
    big_pc_ecef = coordinate_transform(big_pc)
    big_pc_label_ecef = np.concatenate((big_pc_ecef, label), axis=1)
    # save big_pc_label.npy file
    np.save('big_pc_label.npy', big_pc_label_ecef)


if __name__ == '__main__':
	main()
