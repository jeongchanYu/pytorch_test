import torch

def l1_loss():
    def func(orig, pred):
        loss = torch.mean(torch.abs(orig - pred), 2)
        loss = torch.sum(loss)
        return loss
    return func