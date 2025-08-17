import torch
def pcc(pred, true):
    pred, true = pred.flatten(), true.flatten()
    pred = pred - pred.mean(); true = true - true.mean()
    num = (pred * true).sum()
    den = torch.sqrt((pred**2).sum() * (true**2).sum()) + 1e-8
    return (num / den).item()
def ccc(pred, true):
    pred, true = pred.flatten(), true.flatten()
    mx, my = pred.mean(), true.mean()
    vx, vy = pred.var(unbiased=False), true.var(unbiased=False)
    cov = ((pred-mx)*(true-my)).mean()
    return (2*cov / (vx+vy+(mx-my)**2 + 1e-8)).item()
def mae(pred, true): return torch.mean(torch.abs(pred-true)).item()
def mse(pred, true): return torch.mean((pred-true)**2).item()
