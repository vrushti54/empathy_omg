import json, numpy as np, torch
from torch.utils.data import Dataset

class OMGDataset(Dataset):
    def __init__(self, split_json, feature_kind="audio_melspec"):
        with open(split_json) as f: self.items = json.load(f)
        self.feature_kind = feature_kind
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        row = self.items[i]
        x = np.load(f"data/features/{self.feature_kind}/{row['id']}.npy")
        if x.ndim == 2:
            x = x.mean(axis=1)  # average over time
        return torch.tensor(x, dtype=torch.float32), torch.tensor([row["target"]], dtype=torch.float32)
