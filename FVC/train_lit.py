import lightning as L

from net import VideoCompressor, Var

# config
training_lambda = 2048

learning_rate = 1e-4

batch_size = 4
total_setp = 2e6
total_eppch = 1e6

single_trainig_step = 2e6



class FVCLit(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = VideoCompressor()
    

    def training_step(self, inputs, idx):
        if self.global_step % 100 == 0:
            self.model.Trainstage(self.global_step)

        ref_frame = Var(inputs[0])
        input_frames = Var(inputs[1])
        

