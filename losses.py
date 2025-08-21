import torch

def ccc_loss(pred, true):
    """
    pred, true: tensors of shape [B] or [B,1]
    Returns 1 - CCC so minimizing this maximizes CCC.
    """
    pred = pred.view(-1)
    true = true.view(-1)
    mx, my = pred.mean(), true.mean()
    vx, vy = pred.var(unbiased=False), true.var(unbiased=False)
    cov = ((pred - mx) * (true - my)).mean()
    ccc = (2 * cov) / (vx + vy + (mx - my)**2 + 1e-8)
    return 1.0 - ccc
