
import argparse, os, json, torch
from torch.utils.data import DataLoader
from utils_mlp import OMGDatasetMLP
from models.mlp import MLP
from metrics import ccc, pcc, mae, mse

def evaluate(model, loader, device):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device).squeeze(-1)
            preds.append(model(x).cpu()); trues.append(y.cpu())
    pred, true = torch.cat(preds), torch.cat(trues)
    return dict(CCC=ccc(pred,true), PCC=pcc(pred,true), MAE=mae(pred,true), MSE=mse(pred,true))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--norm", default="z", choices=["none","z"])
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = OMGDatasetMLP("data/splits/test.json", norm=args.norm)
    ld = DataLoader(ds, batch_size=64, shuffle=False)

    in_dim = ds.dim
    model = MLP(in_dim, hidden=[512,256,128], dropout=0.3).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    m = evaluate(model, ld, device)
    os.makedirs("results", exist_ok=True)
    out = "results/test_mlp_ccc_audio_melspec.json"
    with open(out, "w") as f: json.dump(m, f, indent=2)
    print("Saved", out, "->", m)

if __name__ == "__main__":
    main()