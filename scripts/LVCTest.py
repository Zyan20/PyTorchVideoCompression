import torch
from torch import nn

import numpy as np
import pandas as pd

import os
from PIL import Image

from metrics import calc_psnr, calc_msssim

"""
Learned Video Compression Test
"""

class AbstractLVCTest():
    def __init__(self) -> None:
        pass

    def intra_encode(self, frame, **kwargs):
        """
        take a frame as input
        encode it with any intra-frame encoder
        return:
            - a encoded frame
            - its bpp
        """
        raise Exception("Not implemented")

    def inter_encode(self, cur_frame, prev_frames, **kwargs):
        """
        take current frame and previous frame(s) as input
        encode it with any inter-frame encoder
        return:
            - a encoded frame
            - its bpp
        """
        raise Exception("Not implemented")
    
    def eval():
        raise Exception("Not implemented")



class LVC_TraditionalIntra_NeuralInter(AbstractLVCTest):
    def __init__(self, seqs_root, im_width, im_height) -> None:
        super().__init__()

        self.im_width = im_width
        self.im_height = im_height

        assert os.path.exists(seqs_root)
        self.seqs_root = seqs_root

        self.pd = None
        self.cur_seq = None
        self.df = None



    def intra_encode(self, seq, frame_id, **kwargs):
        """
        - input
            - seq: name of sequence
            - frame: frame_id
        - return
            - image: np.ndarray, non-normalized
            - bpp
            - psnr
            - msssim
        """
        compressed_img_folder = os.path.join(self.seqs_root, seq, "compressed")
        uncompressed_img_folder = os.path.join(self.seqs_root, seq, "uncompressed")
        report_txt = os.path.join(self.seqs_root, seq, "out", "report.txt")

        if self.cur_seq != seq:
            self.cur_seq = seq
            self.df = pd.read_csv(report_txt)
            self.df = self.df.map(lambda x: x.strip() if isinstance(x, str) else x)


        frame_info = self.df.iloc[frame_id]

        frame_type = frame_info[" Type"]
        if not (frame_type == "I-SLICE" or frame_type == "i-SLICE"):
            raise Exception(f"the {frame_id} frame is not IFrame")
        
        # get bpp
        bpp = frame_info[" Bits"] / (self.im_width * self.im_height)

        

        compressed_image = Image.open(
            os.path.join(compressed_img_folder, "im{:04d}.png".format(frame_id + 1))
        )
        compressed_image = np.array(compressed_image)

        uncompressed_image = Image.open(
            os.path.join(uncompressed_img_folder, "im{:04d}.png".format(frame_id + 1))
        )
        uncompressed_image = np.array(uncompressed_image)

        psnr = calc_psnr(compressed_image, uncompressed_image)
        mssim = calc_msssim(compressed_image, uncompressed_image)

        print("seq: {}, ref: im{:04d}.png".format(seq, (frame_id + 1)))

        return compressed_image, bpp, psnr, mssim



    def inter_encode(self, seq, cur_frame_id, **kwargs):
        compressed_folder = os.path.join(self.seqs_root, seq, "compressed")
        uncompressed_folder = os.path.join(self.seqs_root, seq, "uncompressed")


        cur_frame = Image.open(
            os.path.join(uncompressed_folder, "im{:04d}.png".format(cur_frame_id + 1))
        )
        cur_frame = np.array(cur_frame)

        ref_frame = Image.open(
            os.path.join(compressed_folder, "im{:04d}.png".format((cur_frame_id // self.gop) * self.gop + 1))
        )
        ref_frame = np.array(ref_frame)

        print(seq)
        print("inp: im{:04d}.png".format(cur_frame_id + 1))
        print("ref: im{:04d}.png".format((cur_frame_id // self.gop) * self.gop + 1))

        recon_image, bpp = self.forward_inter_model(cur_frame, ref_frame)

        psnr = calc_psnr(recon_image, cur_frame)
        msssim = calc_msssim(recon_image, cur_frame)

        return recon_image, bpp, psnr, msssim



    def forward_inter_model(self, cur_frame: np.ndarray, ref_frames: np.ndarray):
        """
        input:
            - cur_frame: current frame, non-normalized
            - ref_frame: ref_frames

        return:
            - recon_image: np.ndarray, non-normalized
            - bpp
        """
        raise Exception("Not implemented")

        
        

if __name__ == "__main__":
    tester = LVC_TraditionalIntra_NeuralInter("./sequences", 416, 240, inter_model = None)
    ret = tester.intra_encode(
        seq = "BasketballPass_416x240_50",
        frame_id = 0, 
    )

    # print(ret)







        

        
                
    

