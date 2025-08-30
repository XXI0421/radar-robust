from dataclasses import dataclass, field

@dataclass
class Config:
    data_root: str = 'data'
    seq_len: int = 10
    out_len: int = 10
    batch: int = 2
    lr: float = 3e-4
    epochs: int = 2
    device: str = 'cuda:0'
    model_name: str = 'ConvLSTM1'   # 可扩展：ConvLSTM2 / UNet
    loss_name: str = 'MSE_CSI'      # 可扩展：BCE / Focal / Combo
    csi_weight: float = 0.5
    csi_thr: float = 0.1
    num_workers: int = 0
    save_dir: str = 'checkpoints'
    log_dir: str = 'runs'
    seed: int = 42

    # 供 wandb 一键修改
    def to_dict(self):
        return self.__dict__