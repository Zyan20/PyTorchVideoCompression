import sys
sys.path.append("./util/dataset")

import torch
from torch.utils.data import DataLoader
import lightning as L

from util.dataset.Vimeo90K import Vimeo90K
from DVCLit import DVCLit

# config
train_lambda = 2048
batch_size = 8
L.seed_everything(3407)


model_module = DVCLit(train_lambda)
dataset = Vimeo90K(
    root = "D:/vimeo_septuplet", split_file="sep_trainlist.txt",
    frames = 2, interval = 2
)

dateloader = DataLoader(dataset, batch_size = batch_size)

trainer = L.Trainer(
    max_epochs = 2000,
    fast_dev_run = True,
    devices = "cpu"
)

trainer.fit(model = model_module, dataset = dateloader)
