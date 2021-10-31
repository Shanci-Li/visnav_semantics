import os
import math
import torch
import numpy as np
import torch.utils.model_zoo as model_zoo

from model.Resnet18 import Resnet18
from model.SegNet import SegNet
from model.FCN8s import FCN
from model.BiSeNet import BiSeNet
from model.BiSeNetV2 import BiSeNetV2
from model.PSPNet.pspnet import PSPNet
from model.DeeplabV3Plus import Deeplabv3plus_res50
from model.FCN_ResNet import FCN_ResNet
from model.DDRNet import DDRNet
from model.HRNet import HighResolutionNet


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

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

    def convert_label(self, tensor):
        new_tensor = torch.zeros(tensor.shape)
        new_tensor[tensor == 1] = 1
        new_tensor[tensor == 2] = 2
        new_tensor[tensor == 3] = 3
        new_tensor[tensor == 6] = 4
        new_tensor[tensor == 9] = 5
        new_tensor[tensor == 17] = 6
        return new_tensor

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_path = self.img_path + '/' + self.file_ls[index] + '.png'
        label_path = self.label_path + '/' + self.file_ls[index] + '.dat'
        # Open image
        img_npy = np.array(Image.open(image_path))[:, :, :3]
        # Transform image to tensor
        img_tensor = self.to_tensor(img_npy)
        # read semantic label
        label = self.convert_label(torch.load(label_path))
        name = self.file_ls[index]

        return img_tensor, label, name

    def __len__(self):
        return self.data_len


def build_dataset_train(root, dataset):
    data_dir = os.path.join(root, dataset)
    train_data_list = os.path.join(data_dir, dataset + '_' + 'train_list.txt')
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')

    return


def build_dataset_test(root, dataset, crop_size, mode='whole', gt=False):
    data_dir = os.path.join(root, dataset)
    inform_data_file = os.path.join('./dataset/inform/', dataset + '_inform.pkl')
    train_data_list = os.path.join(data_dir, dataset + '_train_list.txt')

    return


def build_model(model_name, num_classes, backbone='resnet50', pretrained=False, out_stride=32, mult_grid=False):
    if model_name == 'FCN':
        model = FCN(num_classes=num_classes)
    elif model_name == 'FCN_ResNet':
        model = FCN_ResNet(num_classes=num_classes, backbone=backbone, out_stride=out_stride, mult_grid=mult_grid)
    elif model_name == 'SegNet':
        model = SegNet(classes=num_classes)
    elif model_name == 'UNet':
        model = UNet(num_classes=num_classes)
    elif model_name == 'BiSeNet':
        model = BiSeNet(num_classes=num_classes, backbone=backbone)
    elif model_name == 'BiSeNetV2':
        model = BiSeNetV2(num_classes=num_classes)
    elif model_name == 'HRNet':
        model = HighResolutionNet(num_classes=num_classes)
    elif model_name == 'Deeplabv3plus_res50':
        model = Deeplabv3plus_res50(num_classes=num_classes, os=out_stride, pretrained=True)
    elif model_name == "DDRNet":
        model = DDRNet(pretrained=True, num_classes=num_classes)
    elif model_name == 'PSPNet_res50':
        model = PSPNet(layers=50, bins=(1, 2, 3, 6), dropout=0.1, num_classes=num_classes, zoom_factor=8, use_ppm=True,
                       pretrained=True)
    elif model_name == 'PSPNet_res101':
        model = PSPNet(layers=101, bins=(1, 2, 3, 6), dropout=0.1, num_classes=num_classes, zoom_factor=8, use_ppm=True,
                       pretrained=True)

    if pretrained:
        checkpoint = model_zoo.load_url(model_urls[backbone])
        model_dict = model.state_dict()
        # print(model_dict)
        # Screen out layers that are not loaded
        pretrained_dict = {'backbone.' + k: v for k, v in checkpoint.items() if 'backbone.' + k in model_dict}
        # Update the structure dictionary for the current network
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model


