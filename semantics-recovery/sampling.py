import random
import numpy as np
import argparse
import matplotlib.pyplot as plt
from skimage import io
from semantics_recovery_adaptive import all_path
import os


def main():

    args = config_parser()
    label_path = args.label_path
    distance_path = args.distance_path
    image_path = args.image_path
    output_path = args.output_path

    if args.single:
        file_ls = ['_'.join(label_path.split('_')[:-1])]
        label_path = '/'.join(label_path.split('/')[:-1])
        distance_path = '/'.join(distance_path.split('/')[:-1])
        image_path = '/'.join(image_path.split('/')[:-1])

    else:
        file_ls, _ = all_path(label_path, filter_list=['.npy'])
        file_ls = np.unique(file_ls).tolist()
        file_ls = random.sample(file_ls, args.sample_nbr)




    for idx, file in enumerate(file_ls):
        print('sampling image {}'.format(idx))
        image = os.path.join(image_path, '{:s}_img.png'.format('/'.join(file.split('/')[len(label_path.split('/')):])))
        raw_image = io.imread(image)[:, :, :3]
        directory = '/'.join(file.split('/')[len(label_path.split('/')):-1])
        semantics_label = np.load(os.path.join(label_path, directory, '{:s}_semantics.npy'.format(file.split('/')[-1])))
        semantics_distance = np.load(os.path.join(distance_path, directory, '{:s}_distance.npy'.format(file.split('/')[-1])))
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
        plt.savefig(output_path + '/{:s}_sample.png'.format(file.split('/')[-1]),
                    bbox_inches='tight', dpi=400)


def config_parser():
    parser = argparse.ArgumentParser(
        description='Semantic label sampling script.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--single', default=False, action='store_true',
                        help='Use sample mode or single mode.')

    parser.add_argument('--label_path', type=str, default=None,  required=True,
                        help='Directory where the semantic labels are in.')

    parser.add_argument('--distance_path', type=str, default=None,  required=True,
                        help='Directory where the distance files are in.')

    parser.add_argument('--image_path', type=str, default=None,  required=True,
                        help='Directory where the raw image are in.')

    parser.add_argument('--output_path', type=str, default=None,  required=True,
                        help='Directory to store sampling output.')

    parser.add_argument('--sample_nbr', type=int, default=100,
                        help='Number of data points to sample.')

    opt = parser.parse_args()

    return opt


if __name__ == '__main__':
    main()
