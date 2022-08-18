import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
# from dataloader.nyu_transform import *
import os
import torchvision
import json
import scipy.io as sio
import cv2
import numpy as np

class NYUDV2Dataset(Dataset):
    """NYUV2D dataset."""

    def __init__(self, csv_file, transform=None):
        self.dir_anno = os.path.join(csv_file, 'annotations', 'test_annotations.json')
        self.transform = transform
        self.root_path = csv_file
        self.A_paths, self.B_paths, self.AB_anno = self.getData()
        self.tx = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((480,640)),
                                    # torchvision.transforms.CenterCrop((228, 304)),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])

        self.lx = torchvision.transforms.Compose([
                                    torchvision.transforms.Resize((480,640)),
                                    # torchvision.transforms.CenterCrop((228, 304)),
                                    torchvision.transforms.ToTensor()
                                    ])

    def getData(self):
        with open(self.dir_anno, 'r') as load_f:
            AB_anno = json.load(load_f)
        if 'dir_AB' in AB_anno[0].keys():
            self.dir_AB = os.path.join(self.root_path, 'test', AB_anno[0]['dir_AB'])
            AB = sio.loadmat(self.dir_AB)
            self.A = AB['rgbs']
            self.B = AB['depths']
            self.depth_normalize = 10.0
        else:
            self.A = None
            self.B = None
        A_list = [os.path.join(self.root_path, 'test', AB_anno[i]['rgb_path']) for i in range(len(AB_anno))]
        B_list = [os.path.join(self.root_path, 'test', AB_anno[i]['depth_path']) for i in range(len(AB_anno))]
        print('Loaded NYUDV2 data!')
        return A_list, B_list, AB_anno

    def __getitem__(self, anno_index):
        A_path = self.A_paths[anno_index]
        B_path = self.B_paths[anno_index]

        A = self.A[anno_index]  # C*W*H
        B = self.B[anno_index] / 10.0  # the max depth is 10m
        A = A.transpose((2, 1, 0))  # H * W * C
        B = B.transpose((1, 0))  # H * W

        A = Image.fromarray(A.astype('uint8')).convert('RGB')

        image = self.tx(A)
        label = B
        # print(label.max().max())

        return image, label

    def __len__(self):
        return len(self.AB_anno)
