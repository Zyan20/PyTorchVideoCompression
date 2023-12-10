import sys
sys.path.append("./util/dataset")

import torch
from torch.utils.data import DataLoader
import lightning as L

from util.dataset.Vimeo90K import Vimeo90K
from DVCLit import DVCLit

# config
train_lambda = 2048
batch_size = 16
L.seed_everything(3407)


model_module = DVCLit(train_lambda)
train_dataset = Vimeo90K(
    root = "/root/autodl-tmp/vimeo_septuplet", split_file="sep_trainlist.txt",
    frames = 2, interval = 2
)

train_dataloader = DataLoader(train_dataset, batch_size = batch_size, num_workers = 8)

trainer = L.Trainer(
    max_epochs = 2000,
    # fast_dev_run = True,
)

trainer.fit(model = model_module, train_dataloaders = train_dataloader)
