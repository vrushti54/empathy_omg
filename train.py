import json, os, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from utils import OMGDataset
from metrics import ccc, pcc, mae, mse
from models.mlp import MLP

def evaluate(model, loader, device):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device).squeeze(-1)
            out = model(x)
            preds.append(out.cpu()); trues.append(y.cpu())
    pred, true = torch.cat(preds), torch.cat(trues)
    return dict(CCC=ccc(pred,true), PCC=pcc(pred,true), MAE=mae(pred,true), MSE=mse(pred,true))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = OMGDataset("data/splits/train.json", feature_kind="audio_melspec")
    val_ds   = OMGDataset("data/splits/val.json",   feature_kind="audio_melspec")
    train_ld = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=64)
    in_dim = train_ds[0][0].shape[0]
    model = MLP(in_dim).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3)
    lossf = nn.L1Loss()
    best = (-1e9, None)
    for ep in range(1, 11):
        model.train()
        for x,y in train_ld:
            x,y = x.to(device), y.to(device).squeeze(-1)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step()
        m = evaluate(model, val_ld, device)
        print(f"Epoch {ep:02d} | CCC={m['CCC']:.4f} PCC={m['PCC']:.4f} MAE={m['MAE']:.4f} MSE={m['MSE']:.4f}")
        if m["CCC"] > best[0]:
            best = (m["CCC"], m)
    os.makedirs("results", exist_ok=True)
    with open("results/mlp_audio_melspec.json","w") as f:
        json.dump(best[1], f, indent=2)
    print("Saved results/mlp_audio_melspec.json")

if __name__ == "__main__":
    main()
