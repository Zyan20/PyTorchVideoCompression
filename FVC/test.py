import os, sys
sys.path.append("/media/zyan/EEFA0486FA044D71/Home/Learning/VideoCompression/PyTorchVideoCompression/scripts")

from net import VideoCompressor, load_model

from LVCTest import LVC_TraditionalIntra_NeuralInter

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image

import numpy as np

from torchvision import transforms
import json


class HEVC_LVCTest(LVC_TraditionalIntra_NeuralInter):
    def __init__(self, 
        seqs_root,
        im_width, 
        im_height,
        gop = 12,
    ) -> None:
        super().__init__(seqs_root, im_width, im_height)

        self.gop = gop

        self.inter_model = VideoCompressor(2048)
        load_model(self.inter_model, "/home/zyan/下载/FVC_pretrain/2048.model")

        self.inter_model.eval()
        self.inter_model.cuda()

        self.seqs = os.listdir(seqs_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    def forward_inter_model(self, cur_frame, ref_frames):
        # norm, add batch dim, to cuda
        input = self.transform(cur_frame.copy()).unsqueeze(0)
        ref =  self.transform(ref_frames.copy()).unsqueeze(0)

        # input, padding = self.pad(input, 64)
        # ref, _ = self.pad(ref, 64)

        input = Variable(input.cuda())
        ref = Variable(ref.cuda())

        recon_image, out = self.inter_model(ref, input)

        bpp = out["bpp"].cpu().detach().numpy()

        # save_image(input, "input.png")
        # save_image(ref, "ref.png")
        # save_image(recon_image, "recon.png")

        # recon_image = self.crop(recon_image, padding)
        recon_image = recon_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)

        recon_image = (recon_image * 255).astype(np.uint8)

        return recon_image, bpp

    def eval(self):
        psnr_arr = []
        bpp_arr = []
        msssim_arr = []

        for seq_cnt, seq in enumerate(self.seqs):
            uncompressed_image_folder = os.path.join(self.seqs_root, seq, "uncompressed")

            tmp_psnr = []
            tmp_bpp = []
            tmp_msssim = []
            
            frame_count = len(os.listdir(uncompressed_image_folder))
            
            for i in range(0, frame_count):
                if i % self.gop == 0:
                    frame_type = "I"

                    image, bpp, psnr, msssim = self.intra_encode(
                        seq = seq,
                        frame_id = i,
                    )
                    tmp_bpp.append(bpp)
                    tmp_psnr.append(psnr)
                    tmp_msssim.append(msssim)


                else:
                    frame_type = "P"

                    image, bpp, psnr, msssim = self.inter_encode(
                        seq = seq,
                        cur_frame_id = i
                    )
                    tmp_bpp.append(bpp)
                    tmp_psnr.append(psnr)
                    tmp_msssim.append(msssim)


                
                print(f"seq: {seq_cnt + 1}/{len(self.seqs)}, frame: {i + 1}/{frame_count}, {frame_type}")
                print(f"bpp: {bpp}, psnr: {psnr}, ms-ssim: {msssim}")
                print()

            psnr_arr.append(np.mean(tmp_psnr))
            msssim_arr.append(np.mean(tmp_msssim))
            bpp_arr.append(np.mean(tmp_bpp))


        avg_psnr = np.mean(psnr_arr)
        avg_bpp = np.mean(bpp_arr)
        avg_msssim = np.mean(msssim_arr)

        info = {
            "psnr": avg_psnr,
            "bpp": avg_bpp,
            "mssims": avg_msssim
        }

        print(info)

        # uvgdrawplt([avg_bpp], [avg_psnr], [avg_msssim], 2e6, testfull = True)


    def pad(self, x, p: int = 2 ** (4 + 3)):
        """
        return x, padding
        """
        h, w = x.size(2), x.size(3)
        new_h = (h + p - 1) // p * p     
        new_w = (w + p - 1) // p * p     
        padding_left = (new_w - w) // 2     
        padding_right = new_w - w - padding_left    
        padding_top = (new_h - h) // 2     
        padding_bottom = new_h - h - padding_top     
        padding = (padding_left, padding_right, padding_top, padding_bottom)    

        x = F.pad(         
            x,         
            padding,         
            mode="constant",         
            value=0,     
        )     
        
        return x, padding   
    
    def crop(self, x, padding):     
        return F.pad(x, tuple(-p for p in padding))



if __name__ == "__main__":
    hevc_test = HEVC_LVCTest(
        seqs_root = "../scripts/seq_hevc_d_384_192",
        im_width = 384,
        im_height = 192,
        gop = 12
    )

    # hevc_test = HEVC_LVCTest(
    #     seqs_root = "/media/zyan/EEFA0486FA044D71/Home/Learning/VideoCompression/PyTorchVideoCompression/scripts/UVG_seq",
    #     im_width = 1920,
    #     im_height = 1024,
    #     gop = 12
    # )

    # image, bpp, psnr, msssim = hevc_test.intra_encode("BasketballPass_416x240_50", 0)
    # print(image.shape, bpp, psnr, msssim)
    # image, bpp, psnr, msssim = hevc_test.inter_encode("BasketballPass_416x240_50", 1)
    # print(image.shape, bpp, psnr, msssim)

    hevc_test.eval()
