import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.zoo import cheng2020_anchor

from PIL import Image
import numpy as np
import math

from pytorch_msssim import ms_ssim

from tqdm import tqdm
from util.benchmark.LVCTest import AbstractLVCTest
from util.metrics import compute_msssim, compute_psnr, mse2psnr
from util.ImageIO import img2torch

# models
from mDVC.DVC import DeepVideoCompressor
from DVC.net import VideoCompressor as huDVC
from FVC.net import VideoCompressor as huFVC

class LVCTestNeural(AbstractLVCTest):
    def __init__(self, seqs_root, intra_quality = 6, gop = 12):
        super().__init__()

        self.intra_model = cheng2020_anchor(quality = intra_quality, pretrained = True).eval().cuda()

        self.inter_model = None

        self.seqs_root = seqs_root
        self.gop = gop

    def intra_encode(self, input_frame: torch.Tensor):
        out_net = self.intra_model(input_frame)
        psnr = compute_psnr(out_net["x_hat"], input_frame)
        _ms_ssim = compute_msssim(out_net["x_hat"], input_frame)
        bpp = self._compute_intra_bpp(out_net)

        return out_net["x_hat"], psnr, _ms_ssim, bpp

    def eval(self):
        bpp_arr = []
        ms_mssim_arr = []
        psnr_arr = []

        for seq in tqdm(os.listdir(self.seqs_root), desc = "sequence"):
            seq_folder = os.path.join(self.seqs_root, seq)

            seq_bpp = []
            seq_psnr = []
            seq_ms_ssim = []

            ref_frame = None

            for i, img in enumerate(
                tqdm(
                    os.listdir(seq_folder), 
                    desc = "frame", leave = False
                )
            ):
                img_path = os.path.join(seq_folder, img)
                x = img2torch(Image.open(img_path)).cuda()

                if i % self.gop == 0:
                    type = "I"
                    compressed_img, psnr, ms_mssim, bpp = self.intra_encode(x)
                
                    ref_frame = compressed_img

                    seq_bpp.append(bpp)
                    seq_psnr.append(psnr)
                    seq_ms_ssim.append(ms_mssim)

                else:
                    type = "P"
                    recon_image, psnr, ms_ssim, bpp = self.inter_encode(ref_frame, x)

                    ref_frame = recon_image

                    seq_bpp.append(bpp)
                    seq_psnr.append(psnr)
                    seq_ms_ssim.append(ms_ssim)

                # print(i, type, bpp, psnr)

            bpp_arr.append(np.mean(seq_bpp))
            psnr_arr.append(np.mean(seq_psnr))
            ms_mssim_arr.append(np.mean(seq_ms_ssim))
        
        avg_bpp = np.mean(bpp_arr)
        avg_psnr = np.mean(psnr_arr)
        avg_ms_ssim = np.mean(ms_mssim_arr)

        print(avg_bpp, avg_psnr, avg_ms_ssim)

    
    def _compute_intra_bpp(self, out_net):
        size = out_net['x_hat'].size()
        num_pixels = size[0] * size[2] * size[3]
        return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
                for likelihoods in out_net['likelihoods'].values()).item()



class mDVCTest(LVCTestNeural):
    def __init__(self, seqs_root, ckpt, intra_quality = 6, gop=12):
        super().__init__(seqs_root, intra_quality, gop)

        self.inter_model = DeepVideoCompressor(motion_dir = "mDVC/data/flow_pretrain_np/")
        self.inter_model.load_state_dict(torch.load(ckpt))
        self.inter_model.eval().cuda()


    def inter_encode(self, ref_frame, input_frame):
        recon_image, out = self.inter_model(ref_frame, input_frame)

        ms_ssim = compute_msssim(recon_image, input_frame)

        bpp = out["bpp"].detach().cpu().item()
        psnr = mse2psnr(out["recon_mse"].detach().cpu().item())

        return recon_image, psnr, ms_ssim, bpp


class huDVCTest(LVCTestNeural):
    def __init__(self, seqs_root, ckpt, intra_quality = 6, gop=12):
        super().__init__(seqs_root, intra_quality, gop)

        self.inter_model = huDVC()
        self.inter_model.load_state_dict(torch.load(ckpt))
        self.inter_model.eval().cuda()


    def inter_encode(self, ref_frame, input_frame):
        out = self.inter_model(input_frame, ref_frame)  # ref the latter
        recon_image = out[0]
        bpp = out[7].item()

        ms_ssim = compute_msssim(recon_image, input_frame)
        psnr = compute_psnr(recon_image, input_frame)

        return recon_image, psnr, ms_ssim, bpp


class huFVCTest(LVCTestNeural):
    def __init__(self, seqs_root, ckpt, intra_quality = 6, gop=12):
        super().__init__(seqs_root, intra_quality, gop)

        self.inter_model = huFVC(2048)
        self.inter_model.load_state_dict(torch.load(ckpt))
        self.inter_model.eval().cuda()


    def inter_encode(self, ref_frame, input_frame):
        recon_image, out = self.inter_model(ref_frame, input_frame)  # ref the latter
        bpp = out["bpp"].item()

        ms_ssim = compute_msssim(recon_image, input_frame)
        psnr = compute_psnr(recon_image, input_frame)

        return recon_image, psnr, ms_ssim, bpp


if __name__ == "__main__":
    data_folder = "D:/HEVC_Sequence/416x240_50/img_384_192"

    mDVCTester = mDVCTest(data_folder, ckpt = "mDVC/ckpt/2.ckpt")
    mDVCTester.eval()

    huDVCTester = huDVCTest(data_folder, ckpt = "DVC/ckpt/2048.model")
    huDVCTester.eval()

    huFVCTester = huFVCTest(data_folder, ckpt = "FVC/ckpt/2048.model")
    huFVCTester.eval()

