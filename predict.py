import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from net import UNet
#from utils import plot_img_and_mask
import numpy as np
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
import torchvision
import os
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from Data_Loader import Images_Dataset_folder
from nyudv2_dataloader import NYUDV2Dataset

def compute_errors(gt, pred):
    thresh = np.maximum((gt / (pred+1e-10)), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred+1e-10)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred+1e-10) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred+1e-10) - np.log10(gt))
    log10 = np.mean(err)

    return  rms, log_rms, d1, d2, d3, abs_rel, sq_rel, log10

if __name__ =='__main__':

    output_path = './output'
    output_path_1 = './output_12' 
    os.makedirs(output_path,exist_ok=True)
    checkpoint_path = './checkpoints/checkpoint.pth'

    cuda = torch.cuda.is_available()
    gpu_ids = [i for i in range(torch.cuda.device_count())]
    TensorType = torch.cuda.FloatTensor if cuda else torch.Tensor

    # testdata_path_img = '/home/linxiu/dataset/nyu/nyu_label/test/images/'
    # testdata_path_dep = '/home/linxiu/dataset/nyu/nyu_label/test/depths/'

    transformI = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((240,320)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])

    transformM = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((480,640)),
                                        torchvision.transforms.ToTensor()
                                        ])

    # test_dataset = Images_Dataset_folder(testdata_path_img, testdata_path_dep, transformI = transformI, transformM = transformM)
    test_dataset = NYUDV2Dataset('./dataset/')
    val_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=cuda)

    #models
    model = UNet(n_channels=3, n_classes=1)
    #model = torch.nn.DataParallel(model, device_ids=[0])

    if cuda:
        model.cuda().eval()

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['vq-net'])
    print('Model loaded from correct')

    val_loss = torch.zeros(8).cuda()
    test_bar = tqdm(val_loader)
    for idx, (rgb,real) in enumerate(test_bar):
        if cuda:
            rgb = rgb.cuda()

        # _, depth = model(rgb)

        '''
        plt.imsave(os.path.join(output_path, '{}_img.png'.format(idx)), (make_grid((rgb /2 +0.5).cpu().data)).numpy().transpose(1, 2, 0))
        plt.imsave(os.path.join(output_path, '{}_fake.png'.format(idx)), (make_grid(depth.cpu().data)).numpy().transpose(1, 2, 0))
        plt.imsave(os.path.join(output_path, '{}_real.png'.format(idx)), (make_grid(real.cpu().data)).numpy().transpose(1, 2, 0))
        plt.imsave(os.path.join(output_path_1, '{}_out_1.png'.format(idx)), (make_grid(out_1.cpu().data)).numpy().transpose(1, 2, 0))
        plt.imsave(os.path.join(output_path_1, '{}_out_2.png'.format(idx)), (make_grid(out_2.cpu().data)).numpy().transpose(1, 2, 0))
		'''
        # depth = torch.nn.functional.interpolate(depth, size = [real.size(1), real.size(2)], mode='bilinear', align_corners=True)
        
        # real = real.numpy().squeeze()
        # depth = depth.detach().cpu().numpy().squeeze()
        rgb = rgb.cpu().numpy().squeeze()

        # real_val = real[45:471, 41:601]
        # depth_val = depth[45:471, 41:601]
        rgb_val = rgb[:, 45:471, 41:601]

        plt.imsave(os.path.join(output_path, '{}_img.png'.format(idx)), (rgb_val /2 + 0.5).transpose(1,2,0))
        # plt.imsave(os.path.join(output_path, '{}_fake.png'.format(idx)), depth_val, cmap='rainbow')
        # plt.imsave(os.path.join(output_path, '{}_real.png'.format(idx)), real_val, cmap='rainbow')
        
        '''
        mask = real_val < 1
        real_val = real_val[mask]
        depth_val = depth_val[mask]

        loss = compute_errors(10 * real_val, 10 * depth_val)
        val_loss += torch.tensor(loss).cuda()
        '''

    dice = val_loss / (idx + 1)
    print('rms:{:.6f} log_rms:{:.6f} d1:{:.6f} d2:{:.6f} d3:{:.6f} abs_rel:{:.6f} sq_rel:{:.6f} log10:{:.6f}'.format(dice[0], dice[1], dice[2], dice[3], dice[4], dice[5], dice[6], dice[7]))
