import open3d as o3d
import open3d.visualization.gui as gui
from open3d.cpu.pybind.utility import Vector3dVector
from semantics_recovery_adaptive_super_speed import *


def display_outlier(cloud: o3d.geometry.PointCloud, ind: list, file_name: str, label: np.ndarray) -> None:
    labels = label.reshape(-1, 1)[ind].squeeze()
    colors = plt.get_cmap("terrain")(labels / 17)

    vis = o3d.visualization.Visualizer()
    vis.create_window('Visualization_{:s}'.format(file_name))
    chosen_cloud = cloud.select_by_index(ind)
    vis.add_geometry(chosen_cloud)
    outlier_cloud = cloud.select_by_index(ind, invert=True)
    vis.add_geometry(outlier_cloud)
    # show removed points in red, remained in gray
    outlier_cloud.paint_uniform_color([1, 0, 0])

    chosen_cloud.colors = Vector3dVector(colors[:, :3])
    # visualization
    vis.run()



_big_pc_label = np.load("big_pc_label.npy")  # [X, 4]
bound_xyz_min = _big_pc_label[:, :3].min(axis=0)  # [3]
bound_xyz_max = _big_pc_label[:, :3].max(axis=0)  # [3]
offset_center = (bound_xyz_max + bound_xyz_min) / 2  # [3]
interval_xyz = bound_xyz_max - bound_xyz_min  # [3]
scale = 1.0
nodata_value = -1.0
file_ls = all_path('scene_coord', filter_list=['.png'])
app = gui.Application.instance
app.initialize()

for idx_dp, file_name in enumerate(file_ls):
    time_start = time.time()
    _sc = np.load('{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
    label = np.load('scene_coord/semantics_label_{:s}.npy'.format(file_name.split('/')[-1]))  # [480, 720, 3]
    _sc_raw_size = _sc.shape  # [H, W, 3]
    _sc = apply_offset(_sc.reshape(-1, 3), offset_center, scale, nodata_value=-1).reshape(
        _sc_raw_size)  # [H, W, 3], numpy array
    _sc_pts = _sc.reshape(-1, 3)

    flag = _sc_pts[:, 1] != nodata_value  # [3]

    selected_pts = _sc_pts[flag]

    selected_pts = Vector3dVector(selected_pts)
    sc_pc = o3d.geometry.PointCloud(points=selected_pts)

    axis_aligned_bounding_box = sc_pc.get_axis_aligned_bounding_box()
    box_points = axis_aligned_bounding_box.get_box_points()
    box_points = np.asarray(box_points)
    print('axis_aligned_bounding_box: {}'.format(box_points))
    nbr_pts_init = len(sc_pc.points)

    sc_cleaned, idx = sc_pc.remove_statistical_outlier(100, 10)
    nbr_pts_remain = len(sc_cleaned.points)
    print('{} points at begining'.format(nbr_pts_init))
    print('{} points after cleaning:'.format(nbr_pts_remain))
    print('{} points removed:'.format(nbr_pts_init - nbr_pts_remain))
    # remove statistical outlier visualization
    display_outlier(sc_pc, idx, file_name, label)

    time_elapsed = time.time() - time_start
    print("Semantics label recovery time: {:.1f}s".format(time_elapsed))
