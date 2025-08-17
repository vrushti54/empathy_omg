import json, numpy as np, torch
from torch.utils.data import Dataset

class OMGSeqDataset(Dataset):
    def __init__(self, split_json, feature_kind="audio_melspec", T=300):
        with open(split_json) as f: self.items = json.load(f)
        self.feature_kind = feature_kind
        self.T = T  # time steps (frames)
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        row = self.items[i]
        M = np.load(f"data/features/{self.feature_kind}/{row['id']}.npy")  # (n_mels, time)
        # pad/trim along time to fixed T
        n_mels, t = M.shape
        if t >= self.T:
            M = M[:, :self.T]
        else:
            pad = np.zeros((n_mels, self.T - t), dtype=M.dtype)
            M = np.concatenate([M, pad], axis=1)
        # return as (T, n_mels) for RNNs
        X = torch.tensor(M.T, dtype=torch.float32)
        y = torch.tensor([row["target"]], dtype=torch.float32)
        return X, y
