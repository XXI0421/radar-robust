import os, torch
from src.config import Config
from src.dataset import build_loaders
from src.model import build_model
from src.loss import LOSSES
from src.trainer import Trainer
from torch.optim.lr_scheduler import ReduceLROnPlateau

cfg = Config()
os.makedirs(cfg.save_dir, exist_ok=True)

import wandb
wandb.init(project="radar-robust", config=cfg.to_dict())

train_loader, val_loader = build_loaders(cfg)
model = build_model(cfg)
loss_fn = LOSSES[cfg.loss_name](cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

trainer = Trainer(cfg, model, train_loader, val_loader, loss_fn, optimizer, scheduler)
trainer.run()