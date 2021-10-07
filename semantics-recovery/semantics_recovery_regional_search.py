import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from skimage import io


def main():
	downsample_rate = 1
	# read point cloud with semantic label data from .npy file
	_big_pc_label = np.load("big_pc_label.npy")

	file_ls = os.listdir('scene_coord')
	file_ls = ['_'.join(item.split('_')[:-1]) for item in file_ls]
	file_ls = np.unique(file_ls).tolist()
	print(file_ls)

	for idx_dp, file_name in tqdm(enumerate(file_ls)):
		# load ray-traced point cloud and convert ECEF wgs84 into LV95
		_sc = np.load('scene_coord/{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
		raw_image = io.imread('scene_coord/{:s}_img.png'.format(file_name))[:, :, :3]
		_sc = _sc[::downsample_rate, ::downsample_rate, :].transpose(2, 0, 1)  # [3, 60, 90]
		sc = _sc.reshape(3, -1).transpose(1, 0)  # [K, 3]
		sc = torch.tensor(sc).float().cuda()  # [K, 3]
		# find the coordinate edge of the matching image
		sc_xyz_max, _ = sc[sc[:, 1] != -1].max(dim=0)
		sc_xyz_min, _ = sc[sc[:, 1] != -1].min(dim=0)
		sc_xyz_max = sc_xyz_max.cpu().numpy()
		sc_xyz_min = sc_xyz_min.cpu().numpy()

		big_pc_label = _big_pc_label[(_big_pc_label[:, 0] <= sc_xyz_max[0]) & (_big_pc_label[:, 1] <= sc_xyz_max[1])
                         & (_big_pc_label[:, 2] <= sc_xyz_max[2]) & (_big_pc_label[:, 0] >= sc_xyz_min[0])
                         & (_big_pc_label[:, 1] >= sc_xyz_min[1]) & (_big_pc_label[:, 2] >= sc_xyz_min[2])]
		big_pc = torch.tensor(big_pc_label[:, :3]).float().cuda()  # [2N, 3]

		# search for cloest point
		# loop over each pixel
		# this step should be optimized for efficiency
		semantics_label = []
		for idx, query_pt in tqdm(enumerate(sc), desc="Iterating over pixels to retrieve semantic labels", ncols=10):
			if query_pt[0] == -1:
				semantics_label.append(-1)
			else:
				qeury2pc_dist = torch.norm(query_pt.view(1, 3) - big_pc, dim=-1, p=2)  # [N] <- [N, 3]
				idx_cloest_pt_in_pc = qeury2pc_dist.argmin().item()  # scalar
				semantics_label.append(big_pc_label[idx_cloest_pt_in_pc, -1])  # discrete label

		semantics_label = np.array(semantics_label)  # [K]
		semantics_label = semantics_label.reshape(_sc.shape[1], _sc.shape[2])  # [60, 90]
		np.save('semantics_label_{:s}'.format(file_name), semantics_label)

	# for idx_dp, file_name in tqdm(enumerate(file_ls)):
	# 	# load ray-traced point cloud and convert ECEF wgs84 into LV95
	# 	_sc = np.load('scene_coord/{:s}_pc.npy'.format(file_name))  # [480, 720, 3]
	# 	raw_image = io.imread('scene_coord/{:s}_img.png'.format(file_name))[:,:,:3]
	# 	_sc = _sc[::downsample_rate, ::downsample_rate, :].transpose(2, 0, 1)  # [3, 60, 90]
	# 	sc = _sc.reshape(3, -1).transpose(1, 0) # [K, 3]
	# 	sc = torch.tensor(sc).float().cuda()  # [K, 3]
	#
	#
	# 	# search for cloest point
	# 	# loop over each pixel
	# 	# this step should be optimized for efficiency
	# 	idx_closest_pt_in_pc = torch.cdist(sc, big_pc, p=2).argmin(dim=1).item()
	# 	semantics_label = []
	# 	semantics_label = _big_pc[idx_closest_pt_in_pc][:, 3]
	#
	# 	semantics_label = np.array(semantics_label)  # [K]
	# 	semantics_label = semantics_label.reshape(_sc.shape[1], _sc.shape[2])  # [60, 90]
	# 	np.save('semantics_label_{:s}'.format(file_name), semantics_label)

		print(np.unique(semantics_label))
		fig, axes = plt.subplots(1, 2)
		axes[0].axis('off')
		axes[0].imshow(raw_image)

		axes[1].axis('off')
		axes[1].imshow(semantics_label)
		plt.savefig('semantics_label_{:s}.png'.format(file_name), bbox_inches='tight', dpi=400)



if __name__ == '__main__':
	main()