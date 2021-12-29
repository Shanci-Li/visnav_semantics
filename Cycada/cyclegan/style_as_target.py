import os
import numpy as np
from skimage import io
import matplotlib.pyplot as plt

rec_idt = False
input_path = '/DataDisk/style_as_target/first_attempt_LHS/train_drone_sim_20/images'
output_path = os.path.join('/'.join(input_path.split('/')[:-1]), 'results')
if not os.path.exists(output_path):
    os.makedirs(output_path)

file_ls = os.listdir(input_path)
image_ls = ['_'.join(file_name.split('_')[:-2]) for file_name in file_ls]
image_ls = np.unique(image_ls)


for idx, image in enumerate(image_ls):
    print('integrate epoch {}'.format(idx))
    if rec_idt:
        fig, axes = plt.subplots(2, 4)
        recA_path = os.path.join(input_path, '{:s}_rec_A.png'.format(image))
        recB_path = os.path.join(input_path, '{:s}_rec_B.png'.format(image))
        idtA_path = os.path.join(input_path, '{:s}_idt_A.png'.format(image))
        idtB_path = os.path.join(input_path, '{:s}_idt_B.png'.format(image))
        recA_image = io.imread(recA_path)
        recB_image = io.imread(recB_path)
        idtA_image = io.imread(idtA_path)
        idtB_image = io.imread(idtB_path)

        axes[0, 2].axis('off')
        axes[0, 2].imshow(recA_image)
        axes[0, 2].set_title("recA")

        axes[1, 2].axis('off')
        axes[1, 2].imshow(recB_image)
        axes[1, 2].set_title("recB")

        axes[0, 3].axis('off')
        axes[0, 3].imshow(idtA_image)
        axes[0, 3].set_title("idtA")

        axes[1, 3].axis('off')
        axes[1, 3].imshow(idtB_image)
        axes[1, 3].set_title("idtB")

    else:
        fig, axes = plt.subplots(2, 2)

    realA_path = os.path.join(input_path, '{:s}_real_A.png'.format(image))
    realB_path = os.path.join(input_path, '{:s}_real_B.png'.format(image))
    fakeA_path = os.path.join(input_path, '{:s}_fake_A.png'.format(image))
    fakeB_path = os.path.join(input_path, '{:s}_fake_B.png'.format(image))

    realA_image = io.imread(realA_path)
    realB_image = io.imread(realB_path)
    fakeA_image = io.imread(fakeA_path)
    fakeB_image = io.imread(fakeB_path)



    axes[0, 0].axis('off')
    axes[0, 0].imshow(realA_image)
    axes[0, 0].set_title("realA")

    axes[1, 0].axis('off')
    axes[1, 0].imshow(realB_image)
    axes[1, 0].set_title("realB")

    axes[0, 1].axis('off')
    axes[0, 1].imshow(fakeB_image)
    axes[0, 1].set_title("fakeB")

    axes[1, 1].axis('off')
    axes[1, 1].imshow(fakeA_image)
    axes[1, 1].set_title("fakeA")

    plt.savefig(output_path + '/{:s}.png'.format(image), bbox_inches='tight', pad_inches=0.1, dpi=600)
