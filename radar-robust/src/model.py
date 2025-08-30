import torch.nn as nn
from .config import Config
import torch

class ConvLSTM1(nn.Module):   # 单层
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.conv = nn.Conv2d(1 + hidden_dim, 4 * hidden_dim, 3, padding=1)
        self.out_conv = nn.Conv2d(hidden_dim, 1, 3, padding=1)

    def forward(self, x, future=0):
        b, t, _, h, w = x.size()
        h_cur = torch.zeros(b, self.hidden_dim, h, w, device=x.device)
        c_cur = torch.zeros_like(h_cur)
        outputs = []
        out = None
        for step in range(t + future):
            xt = x[:, step] if step < t else out
            combined = torch.cat([xt, h_cur], dim=1)
            gates = self.conv(combined)
            i, f, o, g = gates.chunk(4, 1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c_cur = f * c_cur + i * g
            h_cur = o * torch.tanh(c_cur)
            out = torch.sigmoid(self.out_conv(h_cur))
            outputs.append(out)
        return torch.stack(outputs, 1)[:, :t + future]

MODELS = {
    'ConvLSTM1': ConvLSTM1,
    # 后续可注册：ConvLSTM2, UNet, Swin, ...
}

def build_model(cfg: Config):
    return MODELS[cfg.model_name]()