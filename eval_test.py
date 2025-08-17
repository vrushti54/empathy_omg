import argparse, json, os, torch
from torch.utils.data import DataLoader

# our helpers
from metrics import ccc, pcc, mae, mse
from utils import OMGDataset                # flat features (for MLP)
from utils_seq import OMGSeqDataset, Pad1D  # sequence features (for LSTM/GRU/CNN1D)

# ----- evaluation util -----
def evaluate(model, loader, device):
    model.eval(); preds=[]; trues=[]
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device).squeeze(-1)
            out = model(x)
            preds.append(out.detach().cpu()); trues.append(y.detach().cpu())
    pred, true = torch.cat(preds), torch.cat(trues)
    return dict(CCC=ccc(pred,true), PCC=pcc(pred,true), MAE=mae(pred,true), MSE=mse(pred,true))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["mlp","lstm","gru","cnn1d"])
    ap.add_argument("--ckpt",  help="override checkpoint path")
    ap.add_argument("--T", type=int, default=128, help="max time steps for sequence models")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # choose dataset + model class + default ckpt
    if args.model == "mlp":
        # flat feature vectors; average over time already handled in utils.OMGDataset
        test_ds = OMGDataset("data/splits/test.json", feature_kind="audio_melspec")
        from models.mlp import MLP
        in_dim = test_ds[0][0].shape[0]
        model = MLP(in_dim).to(device)
        ckpt_default = "models/ckpt_mlp_best.pt"
        test_ld = DataLoader(test_ds, batch_size=64)

    else:
        # sequence models
        test_ds = OMGSeqDataset("data/splits/test.json", feature_kind="audio_melspec", T=args.T)
        if args.model == "lstm":
            from models.lstm import LSTMReg as SeqModel
            ckpt_default = "models/ckpt_lstm_best.pt"
        elif args.model == "gru":
            from models.gru import GRUReg as SeqModel
            ckpt_default = "models/ckpt_gru_best.pt"
        else:
            from models.cnn1d import CNN1DReg as SeqModel
            ckpt_default = "models/ckpt_cnn1d_best.pt"

        in_dim = test_ds.dim
        model = SeqModel(in_dim).to(device)
        test_ld = DataLoader(test_ds, batch_size=64, collate_fn=Pad1D())

    ckpt_path = args.ckpt or ckpt_default
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    m = evaluate(model, test_ld, device)
    os.makedirs("results", exist_ok=True)
    out = f"results/test_{args.model}_audio_melspec.json"
    with open(out, "w") as f: json.dump(m, f, indent=2)
    print("Saved", out, "->", m)

if __name__ == "__main__":
    main()
