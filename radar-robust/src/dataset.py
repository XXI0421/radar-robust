import glob, os, cv2, numpy as np, torch
from torch.utils.data import Dataset
from .config import Config

class RadarDataset(Dataset):
    def __init__(self, cfg: Config, split='train'):
        self.cfg = cfg
        self.samples = []
        dirs = sorted(glob.glob(os.path.join(cfg.data_root, split, '*')))
        for d in dirs:
            files = sorted(glob.glob(os.path.join(d, '*.png')))
            need = cfg.seq_len + cfg.out_len
            if len(files) >= need:
                for i in range(len(files) - need + 1):
                    self.samples.append(files[i:i + need])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        paths = self.samples[idx]
        imgs = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in paths]
        imgs = [img for img in imgs if img is not None]
        x = np.stack(imgs, 0).astype(np.float32) / 255.0
        x = torch.from_numpy(x).unsqueeze(1)  # (T,1,H,W)
        # 扩展：这里可插入更多增广
        return x[:self.cfg.seq_len], x[self.cfg.seq_len:self.cfg.seq_len + self.cfg.out_len]

def build_loaders(cfg: Config):
    train_ds = RadarDataset(cfg, 'train')
    val_ds   = RadarDataset(cfg, 'test')
    train_loader = torch.utils.data.DataLoader(
        train_ds, cfg.batch, shuffle=True, num_workers=cfg.num_workers)
    val_loader = torch.utils.data.DataLoader(
        val_ds, cfg.batch, shuffle=False, num_workers=cfg.num_workers)
    return train_loader, val_loader