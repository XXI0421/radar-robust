import torch.nn as nn
from .metric import csi
from .config import Config         

class MSE_CSI(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.weight = cfg.csi_weight
        self.thr = cfg.csi_thr

    def forward(self, pred, tgt):
        mse = nn.MSELoss()(pred, tgt)
        csi_score = csi(pred, tgt, thr=self.thr)
        return mse + self.weight * (1 - csi_score)

LOSSES = {'MSE_CSI': MSE_CSI}