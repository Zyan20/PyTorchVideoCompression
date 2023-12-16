from mDVC.DVCLit import DVCLit
from mDVC.DVC import DeepVideoCompressor
import torch

model = DVCLit.load_from_checkpoint("mDVC/ckpt/epoch=83-step=1356852.ckpt")
torch.save(model.model.state_dict(), "mDVC/ckpt/6.ckpt")
