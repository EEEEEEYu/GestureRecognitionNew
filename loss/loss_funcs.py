import torch.nn.functional as F

def cross_entropy_loss(pred, gt, label_smoothing=0.0):
    return F.cross_entropy(pred, gt, label_smoothing=label_smoothing)