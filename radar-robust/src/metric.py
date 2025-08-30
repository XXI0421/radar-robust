import torch

def csi(pred, gt, thr=0.5):
    p = pred > thr
    g = gt > thr
    inter = (p & g).float().sum((2, 3, 4))
    union = (p | g).float().sum((2, 3, 4))
    return (inter / (union + 1e-6)).mean()