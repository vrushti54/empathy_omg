import os, json, argparse
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader

from models.mlp import MLP
from utils_mlp import OMGDatasetMLP
from utils_train import EarlyStopper
from metrics import ccc, pcc, mae, mse
from losses import ccc_loss

def evaluate(model, loader, device):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device); y = y.to(device).squeeze(-1)
            out = model(x)
            preds.append(out.detach().cpu()); trues.append(y.detach().cpu())
    pred, true = torch.cat(preds), torch.cat(trues)
    return dict(CCC=ccc(pred,true), PCC=pcc(pred,true), MAE=mae(pred,true), MSE=mse(pred,true))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--norm", choices=["none","z"], default="z", help="per-frequency z-norm over time before pooling")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=0.5, help="mix: alpha*CCC + (1-alpha)*MAE")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = OMGDatasetMLP("data/splits/train.json", norm=args.norm)
    val_ds   = OMGDatasetMLP("data/splits/val.json",   norm=args.norm)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_ld   = DataLoader(val_ds, batch_size=256)

    in_dim = train_ds[0][0].shape[0]
    model = MLP(in_dim, hidden=[512,256,128], dropout=0.35).to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=2)
    stopper = EarlyStopper(patience=6, mode="max", min_delta=1e-5)

    best_metric = -1e9
    best_report = None
    os.makedirs("models", exist_ok=True)
    ckpt_path = "models/ckpt_mlp_ccc_best.pt"

    l1 = nn.L1Loss()

    for ep in range(1, args.epochs+1):
        model.train()
        for xb, yb in train_ld:
            xb, yb = xb.to(device), yb.to(device).squeeze(-1)
            opt.zero_grad()
            pred = model(xb)
            loss = args.alpha*ccc_loss(pred, yb) + (1.0-args.alpha)*l1(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        m = evaluate(model, val_ld, device)
        sch.step(m["CCC"])
        print(f"Epoch {ep:02d} | LR={opt.param_groups[0]['lr']:.5f} | CCC={m['CCC']:.4f} PCC={m['PCC']:.4f} MAE={m['MAE']:.4f} MSE={m['MSE']:.4f}")

        if m["CCC"] > best_metric:
            best_metric = m["CCC"]; best_report = m
            torch.save(model.state_dict(), ckpt_path)
        if stopper.step(m["CCC"]):
            print('Early stopping triggered.'); break

    os.makedirs("results", exist_ok=True)
    with open("results/mlp_ccc_audio_melspec.json","w") as f:
        json.dump(best_report, f, indent=2)
    print("Saved best ckpt:", ckpt_path)
    print("Saved results/mlp_ccc_audio_melspec.json")

if __name__ == "__main__":
    main()
