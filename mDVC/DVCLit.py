import torch
from torch import nn
from torch.autograd import Variable

import lightning as L

from DVC import DeepVideoCompressor

import math

def Var(x):
    return Variable(x.cuda())


class DVCLit(L.LightningModule):
    def __init__(self, 
        train_lambda = 2048
    ):
        super().__init__()

        self.model = DeepVideoCompressor()
    
    def training_step(self, batch: torch.Tensor, idx):
        B, T, C, H, W = batch.shape

        ref_frame = Var(batch[:, 0, ...])    # take the first frame as ref frame at the begining

        for i in range(1, T):
            input_frame = Var(batch[: i, ...])

            recon_frame, likelihoods = self.model(ref_frame, input_frame)

            # for mulit-frame stage
            # ref_frame = recon_frame

            

    def get_bpp(self, likelihoods, num_pixels):
        bpp_mv_prior = self._calc_bpp(likelihoods["p_mv_prior"], num_pixels)
        bpp_mv = self._calc_bpp(likelihoods["p_mv"], num_pixels)
        bpp_res_prior = self._calc_bpp(likelihoods["p_res_prior"], num_pixels)
        bpp_res = self._calc_bpp(likelihoods["p_res"], num_pixels)

        bpp = bpp_mv_prior + bpp_mv + bpp_res_prior, bpp_res

        return bpp


    def _calc_bpp(self, likelihoods, num_pixels):
        return torch.sum(torch.clamp(-1.0 * torch.log(likelihoods + 1e-5) / math.log(2.0), 0, 50)) / num_pixels





