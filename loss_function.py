import torch

def l1_loss(orig, pred):
    loss = torch.mean(orig - pred, 2)
    loss = torch.sum(loss)
    return loss