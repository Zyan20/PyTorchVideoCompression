import math

import lightning as L

import torch
from torch import optim
from torch.utils.data import DataLoader

from net import VideoCompressor, Var
from util.dataset.Vimeo90K import Vimeo90K

# config
train_lambda = 2048

learning_rate = 1e-4

batch_size = 4
total_setp = 500000
total_epoch = 1e5

single_trainig_step = 2e6


calc_step = 10
log_step = 100


class FVCLit(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = VideoCompressor(train_lambda)

        self.calc_cnt = 0
        self.sum_out = {}

    def training_step(self, sequence, idx):
        # progressive training
        if self.global_step % 100 == 0:
            self.model.Trainstage(self.global_step)


        # for each input frame
        B, T, C, H, W = sequence.size()

        ref_frame = Var(sequence[:, 0, ...])
        
        seq_out = {}

        for i in range(1, T):
            input_frame = Var(sequence[:, i, ...])   # take i-th frame along batch dim

            # frame weight
            self.model.true_lambda = self.model.train_lambda * (i * 2) / (T + 3)

            recon_image, out = self.model(ref_frame, input_frame)

            # take encoded P frame as reference frame
            ref_frame = recon_image

            # mean
            for key, value in out.items():
                if key in seq_out:
                    seq_out[key] += torch.mean(value)
                else:
                    seq_out[key] = torch.mean(value)

        for key in out:
            seq_out[key] /= T

        
        # log
        if self.global_step % self.calc_cnt == 0:
            self.calc_cnt += 1
            for key in seq_out:
                seq_out[key] = seq_out[key].cpu().detach().numpy()

                seq_out["psnr"] = self._MSE2PSNR(seq_out["mse_loss"])
                seq_out["psnr_align"] = self._MSE2PSNR(seq_out["align_loss"])

            for key, value in seq_out.items():
                if key in self.sum_out:
                    self.sum_out[key] += value
                else:
                    self.sum_out[key] = value
        
        if idx % log_step == 0 and self.calc_cnt > 0:

            lr_scheduler = self.lr_schedulers()
            lr_scheduler.step()


            for key in self.sum_out:
                self.sum_out[key] /= self.calc_cnt
            self.log_dict(self.sum_out)

            self.sum_out = {}
            self.calc_cnt = 0

        return seq_out["rd_loss"]
    

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr = learning_rate, betas = [0.9, 0.999])

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer = optimizer,
            T_0 = 20,
            T_mult = 2,
            eta_min = 1e-7
        )

        return [optimizer], [lr_scheduler]


    def _MSE2PSNR(self, mse):
        return 10 * math.log10(1.0 / (mse))



if __name__ == "__main__":
    L.seed_everything(3407)

    dataset = Vimeo90K(
        root = "", split_file="",
        frames = 7
    )

    dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

    model_module = FVCLit()

    trainer = L.Trainer(
        max_epochs = total_epoch,
    )

    trainer.fit(model = model_module, train_dataloaders = dataloader)





        

