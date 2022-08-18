import torch
import torch.nn.functional as F
import numpy as np
from Data_Loader import Images_Dataset, Images_Dataset_folder
from tqdm import tqdm

def compute_errors(gt, pred):
    thresh = np.maximum((gt / (pred + 1e-10)), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred + 1e-10)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred + 1e-10) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred + 1e-10) - np.log10(gt))
    log10 = np.mean(err)

    return  rms, log_rms, d1, d2, d3, abs_rel, sq_rel, log10

def eval_net(net, val_loader, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    val_loss = torch.zeros(8).cuda()

    eval_bar = tqdm(val_loader)
    for batch_idx, (rgb, depth) in enumerate(eval_bar):
        if gpu:
            rgb = rgb.cuda()
        _, coarse_output = net(rgb)

        depth = depth.numpy().squeeze()
        coarse_output = coarse_output.detach().cpu().numpy().squeeze()

        loss = compute_errors(10 * depth, 10 * coarse_output)
        val_loss += torch.tensor(loss).cuda()

    val_loss_mean = val_loss / (batch_idx + 1)

    return val_loss_mean
    '''
    print('{:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(epoch,
        coarse_validation_loss, delta1_accuracy, rmse_linear_loss,
        abs_relative_difference_loss, squared_relative_difference_loss))
    '''
