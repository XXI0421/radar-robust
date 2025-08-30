import os, torch, wandb
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .config import Config
from src.metric import csi

class Trainer:
    def __init__(self, cfg: Config, model, train_loader, val_loader, loss_fn, optimizer, scheduler):
        self.cfg = cfg
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.writer = SummaryWriter(cfg.log_dir)
        self.best_csi = 0.0

    def run(self):
        for epoch in range(self.cfg.epochs):
            self._train_epoch(epoch)
            val_loss, val_csi = self._val_epoch(epoch)
            self.scheduler.step(val_csi)
            if val_csi > self.best_csi:
                self.best_csi = val_csi
                torch.save(self.model.state_dict(), os.path.join(self.cfg.save_dir, 'best.pth'))
        self.writer.close()

    def _train_epoch(self, epoch):
        self.model.train()
        running_loss, running_csi = 0.0, 0.0
        for x, y in tqdm(self.train_loader, desc=f'Train {epoch}'):
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)
            out = self.model(x)
            loss = self.loss_fn(out, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            csi_score = csi(out, y, thr=self.cfg.csi_thr)   
            running_loss += loss.item()
            running_csi  += csi_score.item()                
        avg_loss = running_loss / len(self.train_loader)
        avg_csi = running_csi / len(self.train_loader)
        self.writer.add_scalar('CSI/train', avg_csi, epoch)
        wandb.log({'train_loss': avg_loss, 'train_csi': avg_csi})

    def _val_epoch(self, epoch):
        self.model.eval()
        running_loss, running_csi = 0.0, 0.0
        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)
                out = self.model(x)
                loss = self.loss_fn(out, y)
                csi_score = csi(out, y, thr=self.cfg.csi_thr)   
                running_loss += loss.item()
                running_csi  += csi_score.item()                
        avg_loss = running_loss / len(self.val_loader)
        avg_csi = running_csi / len(self.val_loader)
        self.writer.add_scalar('CSI/val', avg_csi, epoch)
        wandb.log({'val_loss': avg_loss, 'val_csi': avg_csi})
        
        return avg_loss, avg_csi