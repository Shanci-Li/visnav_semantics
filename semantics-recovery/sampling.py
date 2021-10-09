import random
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from skimage import io
import os


def all_path(dirname: str, filter_list: list) -> list:
    """
    extract all the path of .npy file from the main dir
    :param filter_list:          file format to extract
    :param dirname:         main dir name
    :return:                file name list with detailed path
    """

    file_path_list = []

    for maindir, subdir, file_name_list in os.walk(dirname):

        # # current main dir
        # print('main dir:', maindir)
        # # current sub dir
        # print('sub dir:', subdir)
        # # all file under current main dir
        # print('file name list:', file_name_list)

        for filename in file_name_list:
            if 'poses' in filename.split('_'):
                continue

            if os.path.splitext(filename)[1] in filter_list:
                path_detail = os.path.join(maindir, '_'.join(filename.split('_')[:-1]))
                file_path_list.append(path_detail)

    return file_path_list


def main():

    # attention!!! the index 6 of "file.split('/')[6:-1]" below is relevant to the folder length of input_path
    # attention!!! the index 6 of "file.split('/')[6:-1]" below is relevant to the folder length of input_path
    # attention!!! the index 6 of "file.split('/')[6:-1]" below is relevant to the folder length of input_path
    # you need change by yourself below
    input_path = '/work/topo/VNAV/Synthetic_Data/EPFL'
    output_path = '/home/shanli/semantics-recovery/sample_result'
    file_ls = all_path(input_path, filter_list=['.npy'])
    file_ls = np.unique(file_ls).tolist()
    file_ls = random.sample(file_ls, k=100)

    for idx, file in enumerate(file_ls):
        print('sampling image {}'.format(idx))
        raw_image = io.imread('{:s}_img.png'.format(file))[:, :, :3]
        # set the folder path of .npy and .png file
        path = '/home/shanli/semantics-recovery/semantics_label/EPFL/'
        # attention!!! the index 6 of "file.split('/')[6:-1]" below is relevant to the folder length of input_path
        # change the index according to the length from root to the folder  
        directory = '/'.join(file.split('/')[6:-1])
        semantics_label = np.load(path + directory + '/semantics_label_{:s}.npy'.format(file.split('/')[-1]))
        semantics_distance = np.load(path + directory + '/semantics_distance_{:s}.npy'.format(file.split('/')[-1]))
        fig, axes = plt.subplots(1, 3)
        axes[0].axis('off')
        axes[0].imshow(raw_image)
        axes[0].set_title("Image")

        axes[1].axis('off')
        axes[1].imshow(semantics_label)
        axes[1].set_title("Semantics")

        axes[2].axis('off')
        im = axes[2].imshow(semantics_distance)
        axes[2].set_title("Cloest point distance")

        from mpl_toolkits.axes_grid1 import make_axes_locatable

        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        plt.colorbar(im, cax=cax)
        plt.savefig(output_path + '/semantics_label_{:s}.png'.format(file.split('/')[-1]),
                    bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    main()
