import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

import lightning as L

from .DVC import DeepVideoCompressor

import math

def Var(x):
    return Variable(x.cuda())

def mse2psnr(mse):
    return 10 * math.log10(1.0 / (mse))
    

class DVCLit(L.LightningModule):
    def __init__(self, 
        train_lambda = 2048,
        motion_dir = "mDVC/data/flow_pretrain_np/",
    ):
        super().__init__()
        self.automatic_optimization = False

        self.model = DeepVideoCompressor()
        self.model.opticFlow._load_Spynet(motion_dir)

        self.warp_weight = 0
        self.traing_lambda = train_lambda

        self.sum_out = {
            "rd_loss": 0
        }

    
    def training_step(self, batch: torch.Tensor, idx):
        B, T, C, H, W = batch.shape

        ref_frame = Var(batch[:, 0, ...])    # take the first frame as ref frame at the begining

        rd_loss = 0

        for i in range(1, T):
            input_frame = Var(batch[:, i, ...])

            recon_frame, out  = self.model(ref_frame, input_frame)

            distortion = out["recon_mse"] + self.warp_weight * (out["ME_mse"] + out["MC_mse"])
            bpp = out["bpp_mv"] + out["bpp_res_prior"] + out["bpp_res"]
            rd_loss += (bpp + self.traing_lambda * distortion)  # todo


        # backward
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(rd_loss)
        
        self.clip_gradients(opt, gradient_clip_val = 0.5, gradient_clip_algorithm = "norm")
        opt.step()
        
        # 
        lr = self.optimizers().optimizer.state_dict()['param_groups'][0]['lr']


        # sch = self.lr_schedulers()
        # sch.step()


        # log
        for key, value in out.items():
            if key in self.sum_out:
                self.sum_out[key] += value.cpu().detach().item()
            else:
                self.sum_out[key] = value.cpu().detach().item()     
        self.sum_out["rd_loss"] += rd_loss.cpu().detach().item()

        if self.global_step % 100 == 0:
            # self.training_stage()   # change state

            for key in self.sum_out:
                self.sum_out[key] /= 100

            self.sum_out["lr"] = lr
            self.sum_out["psnr"] = mse2psnr(self.sum_out["recon_mse"])
            self.sum_out["ME_psnr"] = mse2psnr(self.sum_out["ME_mse"])
            self.sum_out["MC_psnr"] = mse2psnr(self.sum_out["MC_mse"])
            self.sum_out["bpp"] = self.sum_out["bpp_mv"] + self.sum_out["bpp_res_prior"] + self.sum_out["bpp_res_prior"]
            self.sum_out["warp_weight"] = self.warp_weight
            
            self.log_dict(self.sum_out)

            self.sum_out = {
                "rd_loss": 0
            }


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = 1e-5)
        # lr_scheduler = optim.lr_scheduler.StepLR(
        #     optimizer = optimizer,
        #     step_size = 50000, gamma = 0.1
        # )

        # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer = optimizer,
        #     T_0 = 2000,
        #     T_mult = 2,
        #     eta_min = 1e-7
        # )

        # return [optimizer], [lr_scheduler]
        return optimizer


    def training_stage(self):
        if self.global_step < 40_000:
            self.warp_weight = 0.1

        else:
            self.warp_weight = 0
