
import json, numpy as np, torch
from torch.utils.data import Dataset

class OMGDatasetMLP(Dataset):
    """
    Convert each mel-spec [n_mels, T] into a single vector:
      norm=='none' -> mean over time         -> [n_mels]
      norm=='z'    -> concat(mean, std)      -> [2 * n_mels]
    """
    def __init__(self, split_json, feature_kind="audio_melspec", norm="none"):
        with open(split_json) as f:
            self.items = json.load(f)
        self.feature_kind = feature_kind
        self.norm = norm  # 'none' or 'z'

        # Infer base mel dimension and expose final input dim
        sample = np.load(f"data/features/{feature_kind}/{self.items[0]['id']}.npy")
        if sample.ndim == 1:
            base_dim = int(sample.shape[0])
            self.dim = base_dim if self.norm == "none" else (base_dim if base_dim != 128 else 256)
        else:
            base_dim = int(sample.shape[0])  # n_mels
            self.dim = base_dim * 2 if self.norm == "z" else base_dim

    def __len__(self):
        return len(self.items)

    def _to_vec(self, M):
        # M: [n_mels, T] or [D]
        if M.ndim == 1:
            return M.astype("float32")
        mean = M.mean(axis=1).astype("float32")
        if self.norm == "z":
            std = M.std(axis=1).astype("float32")
            return np.concatenate([mean, std], axis=0)
        return mean

    def __getitem__(self, idx):
        r = self.items[idx]
        M = np.load(f"data/features/{self.feature_kind}/{r['id']}.npy")
        x = self._to_vec(M)
        y = np.array([float(r["target"])], dtype="float32")
        return torch.tensor(x), torch.tensor(y)
