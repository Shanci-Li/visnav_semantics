import os
import cv2
import torch
import numpy as np
import albumentations as A
import torch.utils.model_zoo as model_zoo

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
from model.SegNet import SegNet
from model.FCN8s import FCN
from model.BiSeNet import BiSeNet
from model.BiSeNetV2 import BiSeNetV2
from model.PSPNet.pspnet import PSPNet
from model.DeeplabV3Plus import Deeplabv3plus_res50
from model.FCN_ResNet import FCN_ResNet
from model.DDRNet import DDRNet
from model.HRNet import HighResolutionNet
from model.UNet import UNet


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class SenmanticData(Dataset):
    def __init__(self, dir_path, augmentation=False):
        """
        Args:
            data_path (string): path to file
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.augmentation = augmentation
        self.to_tensor = T.ToTensor()
        # Read the file
        self.img_path = dir_path + '/rgb'
        self.label_path = dir_path + '/semantics'
        self.file_ls = os.listdir(self.img_path)
        self.file_ls = [file.split('.')[0] for file in self.file_ls]
        # Calculate len
        self.data_len = len(self.file_ls)
        if self.augmentation:
            # image augmentation
            self.transform = A.Compose([
                A.OneOf([
                    A.RandomGamma(gamma_limit=(60, 120), p=0.5),
                    A.ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.GaussNoise (var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5)
                ]),
                A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=0.9),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20,
                                   interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, p=1),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
            ])
        else:
            self.transform = A.Compose([
                A.CLAHE(clip_limit=4.0, tile_grid_size=(4, 4), p=1.0),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, p=1.0)
            ])


    def convert_label(self, tensor):
        new_tensor = torch.zeros(tensor.shape)
        new_tensor[tensor == 1] = 1
        new_tensor[tensor == 2] = 1
        new_tensor[tensor == 3] = 2
        new_tensor[tensor == 6] = 3
        new_tensor[tensor == 9] = 4
        new_tensor[tensor == 17] = 5
        return new_tensor

    def __getitem__(self, index):
        # Get image name from the pandas df
        image_path = self.img_path + '/' + self.file_ls[index] + '.png'
        label_path = self.label_path + '/' + self.file_ls[index] + '.npy'
        # Open image
        img_npy = np.array(Image.open(image_path))[:, :, :3]

        # read semantic label
        label = np.load(label_path)

        transformed = self.transform(image=img_npy, mask=label)
        img = transformed['image']
        mask = transformed['mask']

        # Transform image to tensor
        img_tensor = self.to_tensor(img)
        mask = self.convert_label(mask)
        name = self.file_ls[index]

        return img_tensor, mask, name

    def __len__(self):
        return self.data_len


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


