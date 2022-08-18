import sys
import os
from optparse import OptionParser
import numpy as np
import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torchvision
from adamp import AdamP

from eval import eval_net
from net import UNet
from Data_Loader import Images_Dataset, Images_Dataset_folder
from torch.utils.data import DataLoader
from tqdm import tqdm
from loss import total_loss
import math
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def adjust_lr(optimizer, epoch, total_number, C, lr):
    lr = lr/2 * (math.cos((math.pi * divmod(epoch, round(total_number/C))[1])/round(total_number/C))+1)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_unet(net,
              epochs=5,
              batch_size=2,
              lr=0.1,
              val_percent=0.3,
              save_cp=True,
              gpu=True,
              NN=0):

    cudnn.benchmark = True
    dir_checkpoint = os.path.join('./checkpoints', datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
    os.makedirs(dir_checkpoint, exist_ok=True)


    '''
    traindata_path_img = '/home/linxiu/dataset/nyu/nyu_train/nyu_train_images/'
    traindata_path_dep = '/home/linxiu/dataset/nyu/nyu_train/nyu_train_depths/'
    '''

    traindata_path_img = '/home/linxiu/dataset/nyu/nyu_label/train/images/'
    traindata_path_dep = '/home/linxiu/dataset/nyu/nyu_label/train/depths/'


    testdata_path_img = '/home/linxiu/dataset/nyu/nyu_label/test/images/'
    testdata_path_dep = '/home/linxiu/dataset/nyu/nyu_label/test/depths/'

    train_dataset = Images_Dataset_folder(traindata_path_img, traindata_path_dep)

    transformI = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((240,320)),
                                        torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                        ])

    transformM = torchvision.transforms.Compose([
                                        torchvision.transforms.Resize((240,320)),
                                        torchvision.transforms.ToTensor()
                                        ])

    test_dataset = Images_Dataset_folder(testdata_path_img, testdata_path_dep, transformI = transformI, transformM = transformM)

    cuda = torch.cuda.is_available()
    gpu_ids = [i for i in range(torch.cuda.device_count())]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=3, pin_memory=cuda)
    val_loader = DataLoader(test_dataset, batch_size=4, num_workers=3, pin_memory=cuda)

    filename = 'epoch_number.txt'

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_dataset),
               len(test_dataset), str(save_cp), str(gpu)))

    N_train = len(train_dataset)

    optimizer = AdamP(net.parameters(), lr = lr, betas=(0.5, 0.9), weight_decay = 0.0005)
    #optimizer = optim.Adam(net.parameters(), lr = lr, betas=(0.5, 0.9), weight_decay = 0.0005)

    if args.load:
        checkpoint = torch.load(args.load)
        net.load_state_dict(checkpoint['vq-net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('Model loaded from {}'.format(args.load))

    criterion = total_loss()
    if gpu:
        criterion = criterion.cuda()

    #schedule = torch.optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.5, last_epoch=-1)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 20, eta_min=0.000001, last_epoch=-1)
    val_mse_min = 1

    for epoch in range(NN,epochs):
        net.train()
        epoch_loss = 0
        train_bar = tqdm(train_loader)

        if (epoch + 1) % 30 ==0:
            print(optimizer.state_dict()['param_groups'][0]['lr'])
        #adjust_lr(optimizer, epoch, 500, 50, lr)

        for idx, (imgs, true_masks) in enumerate(train_bar):
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            vq_loss, masks_pred = net(imgs)

            #masks_pred_new = ((255 * masks_pred).ceil())/255

            mask = true_masks > 0.01

            train_loss = criterion(masks_pred, true_masks, mask.to(torch.bool))
            loss = train_loss + vq_loss

            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_bar.set_description('Epoch{:^4d}  loss: {:.6f}'.format(epoch+1, loss))

        schedule.step()

        if 1:
            dice = eval_net(net, val_loader, gpu)
            if val_mse_min > dice[0]:
                torch.save({'vq-net':net.state_dict(),'optimizer':optimizer.state_dict()}, os.path.join(dir_checkpoint, 'checkpoint_min.pth'))
                val_mse_min = dice[0]
        if save_cp:
            torch.save({'vq-net':net.state_dict(),'optimizer':optimizer.state_dict()}, os.path.join(dir_checkpoint, 'checkpoint_epoch.pth'))
            with open(filename,'a') as f:
                f.write('{:^4d}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}  {:.6f}\r\n'.format(epoch, epoch_loss / (idx + 1), dice[0], dice[1], dice[2], dice[3], dice[4], dice[5], dice[6], dice[7]))
            print('Loss: {:.6f} rms:{:.6f} log_rms:{:.6f} d1:{:.6f} d2:{:.6f} d3:{:.6f} abs_rel:{:.6f} sq_rel:{:.6f} log10:{:.6f}'.format(epoch_loss / (idx + 1),dice[0], dice[1], dice[2], dice[3], dice[4], dice[5], dice[6], dice[7]))



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=200, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.00015,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-t', '--iter', default = 10, type = 'int', help = 'number of iter')
    parser.add_option('-N', '--epoch_begin', default = 0, type = 'int', help = 'number of epoch begining')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    unet = UNet(n_channels=3, n_classes=1)

    set_seed(11)

    if args.gpu:
        unet.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    print(get_parameter_number(unet))

    try:
        train_unet(net=unet,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  NN = args.epoch_begin)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
