# -*- coding: utf-8 -*-
import imageio
import random
import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
_DEFAULT_MU = [ .5, .5, .5]
_DEFAULT_SIGMA = [ .5, .5, .5]

DEFAULT_TRANSFORM = transforms.Compose([
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomCrop(256, pad_if_needed=True),
    transforms.ToTensor(),
    transforms.Normalize(_DEFAULT_MU, _DEFAULT_SIGMA),
])
# DEFAULT_TRANSFORM = transforms.Compose([
#     transforms.ToTensor()
# ])

class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, path, transform, limit=np.inf):
        super().__init__(path, transform=transform)
        self.limit = limit

    def __len__(self):
        length = super().__len__()
        return min(length, self.limit)


class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):

        # if transform is None:
        #     transform = DEFAULT_TRANSFORM

        super().__init__(
            customData(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )
def default_loader(path):
    try:
        img_list = imageio.mimread(path)
        return img_list
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class customData(Dataset):
    def __init__(self,  path, transform, limit=np.inf, loader=default_loader):
        self.img_name = [os.path.join(path,i) for i in os.listdir(path)]
        self.limit = limit
        self.loader = loader

    def __len__(self):
        return min(len(self.img_name),self.limit)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img = self.loader(img_name)
        img = torch.Tensor(img)
        img = img.div_(127.5).sub_(1)
        img = img.permute([3,0,1,2])

        return img

class DataLoader1(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):
        # 以前没有进行归一化
        if transform is None:
            transform = DEFAULT_TRANSFORM

        super().__init__(
            ImageFolder(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )

#读取灰度图像
class DataLoader2(torch.utils.data.DataLoader):

    def __init__(self, path, transform=None, limit=np.inf, shuffle=True,
                 num_workers=8, batch_size=4, *args, **kwargs):

        # if transform is None:
        #     transform = DEFAULT_TRANSFORM

        super().__init__(
            gary(path, transform, limit),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            *args,
            **kwargs
        )
def default_loader1(path):
    try:
        img = imageio.imread(path)
        return img
    except:
        print("Cannot read image: {}".format(path))

# define your Dataset. Assume each line in your .txt file is [name/tab/label], for example:0001.jpg 1
class gary(Dataset):
    def __init__(self,  path, transform, limit=np.inf, loader=default_loader1):
        self.img_name = [os.path.join(path,i) for i in os.listdir(path)]
        self.limit = limit
        self.loader = loader

    def __len__(self):
        return min(len(self.img_name),self.limit)

    def __getitem__(self, item):
        img_name = self.img_name[item]
        img = self.loader(img_name)
        img = torch.Tensor(img)
        img = img.div_(127.5).sub_(1)

        return img