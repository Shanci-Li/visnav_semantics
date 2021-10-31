import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms as T

PROJECT_DIR = os.path.abspath(os.path.join(os.getcwd()))
sys.path.insert(0, PROJECT_DIR)


class SenmanticData(Dataset):
    def __init__(self, dir_path):
        """
        Args:
            data_path (string): path to file
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.to_tensor = T.ToTensor()
        # Read the file
        self.img_path = dir_path + '/rgb'
        self.label_path = dir_path + '/semantics'
        self.file_ls = os.listdir(self.img_path)
        self.file_ls = [file.split('.')[0] for file in self.file_ls]
        # Calculate len
        self.data_len = len(self.file_ls)

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_path = self.img_path + '/' + self.file_ls[index] + '.png'
        label_path = self.label_path + '/' + self.file_ls[index] + '.dat'
        # Open image
        img_npy = np.array(Image.open(image_path))[:, :, :3]
        # Transform image to tensor
        img_tensor = self.to_tensor(img_npy)
        # read semantic label
        label = torch.load(label_path)
        name = self.file_ls[index]

        return img_tensor, label, name

    def __len__(self):
        return self.data_len


train_set = SenmanticData('./datasets/EPFL/val_real')
val_set = SenmanticData('./datasets/EPFL/val_translated')
test_set = SenmanticData('./datasets/EPFL/val_drone_sim')

train_data = DataLoader(dataset=train_set, batch_size=4, shuffle=True, num_workers=12)
for i, batch in enumerate(train_data):
    # for img, label, size, name in batch:
    #     print(size)
    #     print(name)
    print(batch[2])

