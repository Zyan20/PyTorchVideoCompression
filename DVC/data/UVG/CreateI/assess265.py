import math
import sys
import os.path
import re
import argparse


from skimage import io
import numpy
import pandas as pd
from scipy import signal, ndimage


import matplotlib.pyplot as plt



def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = numpy.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = numpy.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(numpy.float64)
    img2 = img2.astype(numpy.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  #bitdepth of image
    C1 = (K1 * L)**2
    C2 = (K2 * L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(window, img1 * img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2 * img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1 * img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = numpy.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = numpy.ones((2, 2)) / 4.0
    im1 = img1.astype(numpy.float64)
    im2 = img2.astype(numpy.float64)
    mssim = numpy.array([])
    mcs = numpy.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = numpy.append(mssim, ssim_map.mean())
        mcs = numpy.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.convolve(im1, downsample_filter, mode='reflect')
        filtered_im2 = ndimage.convolve(im2, downsample_filter, mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (numpy.prod(mcs[0:level - 1]**weight[0:level - 1]) * (mssim[level - 1]**weight[level - 1]))


def psnr(ref, target):
    diff = ref/255.0 - target/255.0
    diff = diff.flatten('C')
    rmse = math.sqrt(numpy.mean(diff**2.))
    return 20 * math.log10(1.0 / (rmse))


def assess():
    parser = argparse.ArgumentParser()

    parser.add_argument("crf", type = int)
    parser.add_argument("sw", type = int, help = "image size width")
    parser.add_argument("sh", type = int, help = "image size height")

    args = parser.parse_args()
    crf, sw, sh = args.crf, args.sw, args.sh

    outputDir = "out/"
    imageDir = "../images"

    bpps = []
    ms_ssims = []
    psnrs = []

    # for every video
    for video in os.listdir(outputDir):
        reportFilePath = os.path.join(
            outputDir,
            video,
            f"report_{video}.txt"
        )

        reportFrame = pd.read_csv(reportFilePath)

        bits_frames = reportFrame[" Bits"].to_numpy(dtype=numpy.float64)
        bpp_frames = bits_frames / (sw * sh)

        psnrArr = []
        ms_ssimArr = []
        bppArr = []

        # every 12 frames, corresponding to keyint
        for i in range(0, len(bits_frames), 12):
            sourceImgPath = os.path.join(outputDir, video, "source", "img{:04d}.png".format(i + 1))
            h265ImgPath = os.path.join(imageDir, video, f"H265L{crf}", "img{:04d}.png".format(i + 1))
            # print(sourceImgPath, h265ImgPath)

            sourceImg = io.imread(sourceImgPath)
            h265Img = io.imread(h265ImgPath)

            psnr_val = psnr(sourceImg, h265Img)
            
            tmpssim = msssim(h265Img[:, :, 0], h265Img[:, :, 0])
            tmpssim += msssim(h265Img[:, :, 1], h265Img[:, :, 1])
            tmpssim += msssim(h265Img[:, :, 2], h265Img[:, :, 2])

            ms_ssim_val = tmpssim / 3.0

            psnrArr.append(psnr_val)
            ms_ssimArr.append(ms_ssim_val)
            bppArr.append(bpp_frames[i])

        info = {
            "video": video,
            "crf": crf,
            "psnr": numpy.array(psnrArr).mean(0),
            "ms-ssim": numpy.array(ms_ssimArr).mean(0),
            "bpp": numpy.array(bppArr).mean(0)
        }
        print(info)
        print()

        psnrs.append(numpy.array(psnrArr).mean(0))
        ms_ssims.append(numpy.array(ms_ssimArr).mean(0))
        bpps.append(numpy.array(bppArr).mean(0))


    print(numpy.array(psnrs).mean(0))
    print(numpy.array(ms_ssims).mean(0))
    print(numpy.array(bpps).mean(0))


if __name__ == "__main__":
    assess()

    
