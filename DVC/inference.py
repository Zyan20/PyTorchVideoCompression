
import os
import argparse
import torch
import cv2
import logging
import numpy as np
from net import *
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
import sys
import math
import json
from dataset import UVGDataSet
from tensorboardX import SummaryWriter
from drawuvg import uvgdrawplt


def Var(x):
    return Variable(x.cuda())

if __name__ == "__main__":

    uvgDataset = UVGDataSet(
        root = "C:/Home/Learning/VideoCompression/PyTorchVideoCompression/DVC/data/UVG/images/",
        filelist = "C:/Home/Learning/VideoCompression/PyTorchVideoCompression/DVC/data/UVG/originalv.txt",
        refdir = "H265L20",
        testfull = True
    )

    uvgDataLoader = DataLoader(
        dataset = uvgDataset, 
        shuffle = False, 
        num_workers = 0, 
        batch_size = 1,
        pin_memory = True
    )

    net = VideoCompressor()
    load_model(net, "examples/example/snapshot/2048.model")


    with torch.no_grad():
        net.cuda()
        net.eval()

        sumbpp = 0
        sumpsnr = 0
        summsssim = 0
        cnt = 0
        
        for id, input in enumerate(uvgDataLoader):
            print("testing : %d/%d"% (id, len(uvgDataLoader)))
            
            (inputImages, refImage, refBpp, refPSNR, refMS_SSIM) = input
            print(inputImages.size())

            seqlen = inputImages.size()[1]
            sumbpp += torch.mean(refBpp).detach().numpy()
            sumpsnr += torch.mean(refPSNR).detach().numpy()
            summsssim += torch.mean(refMS_SSIM).detach().numpy()
            cnt += 1

            # 对于序列中的每一张图片
            for i in range(seqlen):
                input_image = inputImages[:, i, :, :, :]
                inputframe, refframe = Var(input_image), Var(refImage)
                clipped_recon_image, mse_loss, warploss, interloss, bpp_feature, bpp_z, bpp_mv, bpp = net(inputframe, refframe)
                sumbpp += torch.mean(bpp).cpu().detach().numpy()
                sumpsnr += torch.mean(10 * (torch.log(1. / mse_loss) / np.log(10))).cpu().detach().numpy()
                summsssim += ms_ssim(clipped_recon_image.cpu().detach(), input_image, data_range=1.0, size_average=True).numpy()
                cnt += 1
                ref_image = clipped_recon_image
            
            break

        sumbpp /= cnt
        sumpsnr /= cnt
        summsssim /= cnt
        log = "UVGdataset : average bpp : %.6lf, average psnr : %.6lf, average msssim: %.6lf\n" % (sumbpp, sumpsnr, summsssim)
        print(log)

        
