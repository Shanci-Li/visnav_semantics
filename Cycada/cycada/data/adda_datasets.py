import os.path
import torch.utils.data

from .data_loader import get_transform_dataset
from ..util import to_tensor_raw
from ..transforms import RandomCrop
from ..transforms import augment_collate
import os
import cv2
import torch
import numpy as np
import albumentations as A

from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms as T


class CycadaData(Dataset):
    # def __init__(self, root, num_cls=19, split='train', remap_labels=True,
    #         transform=None, target_transform=None):

    def __init__(self, root, styled=False, augmentation=False):
        """
        Args:
            root (string): path to file
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.augmentation = augmentation
        self.to_tensor = T.ToTensor()
        # Read the file
        if styled:
            self.img_path = root + '/styled_as_target'
        else:
            self.img_path = root + '/rgb'
        self.label_path = root + '/semantics'
        self.file_ls = os.listdir(self.img_path)
        self.file_ls = [file.split('.')[0] for file in self.file_ls]
        # Calculate len
        self.data_len = len(self.file_ls)

        if self.augmentation:
            # image augmentation
            self.transform = A.Compose([
                A.OneOf([
                    A.RandomGamma(gamma_limit=(60, 120), p=0.5),
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, always_apply=False, p=0.5),
                    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5)
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
        mask = self.convert_label(mask).long()
        name = self.file_ls[index]

        return img_tensor, mask, name

    def __len__(self):
        return self.data_len


class AddaDataLoader(object):
    def __init__(self, args):
        self.downscale = args.downscale
        self.crop_size = args.crop_size
        self.half_crop = args.half_crop
        self.batch_size = args.batch_size
        self.shuffle = args.shuffle
        self.num_workers = args.num_workers
        self.gpu = args.gpu
        assert len(args.datafolder)==2, 'Requires two datasets: source, target'
        sourcedir = os.path.join(args.rootdir, args.dataset,args.datafolder[0])
        targetdir = os.path.join(args.rootdir, args.dataset,args.datafolder[1])

        # load scr and tgt set
        self.source = CycadaData(sourcedir, styled=True, augmentation=args.augmentation)
        self.target = CycadaData(targetdir, styled=False, augmentation=args.augmentation)

        # self.source = get_transform_dataset(self.dataset[0], sourcedir,
        #         net_transform, downscale)
        # self.target = get_transform_dataset(self.dataset[1], targetdir,
        #         net_transform, downscale)
        print('Source length:', len(self.source), 'Target length:', len(self.target))
        self.n = max(len(self.source), len(self.target)) # make sure you see all images
        self.num = 0
        self.set_loader_src()
        self.set_loader_tgt()

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.num % len(self.iters_src) == 0:
            print('restarting source dataset')
            self.set_loader_src()
        if self.num % len(self.iters_tgt) == 0:
            print('restarting target dataset')
            self.set_loader_tgt()

        img_src, label_src, _ = next(self.iters_src)
        img_tgt, label_tgt, _ = next(self.iters_tgt)

        if torch.cuda.is_available():
            img_src = img_src.to('cuda:' + self.gpu)
            label_src = label_src.to('cuda:' + self.gpu).long()
            img_tgt = img_tgt.to('cuda:' + self.gpu)
            label_tgt = label_tgt.to('cuda:' + self.gpu).long()
            
        self.num += 1
        return img_src, img_tgt, label_src, label_tgt


    def __len__(self):
        return min(len(self.source), len(self.target))

    def set_loader_src(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        if self.crop_size is not None:
            collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
                    halfcrop=self.half_crop, flip=True)
        else:
            collate_fn=torch.utils.data.dataloader.default_collate
        self.loader_src = torch.utils.data.DataLoader(self.source, 
                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True, drop_last=True)
        self.iters_src = iter(self.loader_src)


    def set_loader_tgt(self):
        batch_size = self.batch_size
        shuffle = self.shuffle
        num_workers = self.num_workers
        if self.crop_size is not None:
            collate_fn = lambda batch: augment_collate(batch, crop=self.crop_size,
                    halfcrop=self.half_crop, flip=True)
        else:
            collate_fn=torch.utils.data.dataloader.default_collate
        self.loader_tgt = torch.utils.data.DataLoader(self.target, 
                batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                collate_fn=collate_fn, pin_memory=True, drop_last=True)
        self.iters_tgt = iter(self.loader_tgt)



