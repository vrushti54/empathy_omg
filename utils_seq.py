import json, os, numpy as np, torch
from torch.utils.data import Dataset

class OMGSeqDataset(Dataset):
    """
    Loads mel-spectrogram .npy (shape [n_mels, T]) and returns sequence
    as [T, n_mels] for RNN/CNN1D models.
    """
    def __init__(self, split_json, feature_kind="audio_melspec", T=None):
        with open(split_json) as f: self.items = json.load(f)
        self.feature_kind = feature_kind
        self.T = T  # optional fixed clip length during __getitem__ (otherwise pad in collate)
        # infer feature dim
        first = np.load(f"data/features/{feature_kind}/{self.items[0]['id']}.npy")
        self.dim = int(first.shape[0])  # n_mels

    def __len__(self): return len(self.items)

    def _load_seq(self, clip_id):
        m = np.load(f"data/features/{self.feature_kind}/{clip_id}.npy")  # [n_mels, T]
        if m.ndim != 2:
            raise ValueError(f"Expected 2D mel-spec, got shape {m.shape}")
        seq = m.T.astype("float32")  # -> [T, n_mels]
        return seq

    def __getitem__(self, i):
        row = self.items[i]
        x = self._load_seq(row["id"])
        if self.T is not None:
            # center-crop or pad to fixed T
            T, D = x.shape
            if T >= self.T:
                start = 0
                x = x[start:start+self.T]
            else:
                pad = np.zeros((self.T, D), dtype="float32")
                pad[:T] = x
                x = pad
        y = np.array([float(row["target"])], dtype="float32")
        return torch.from_numpy(x), torch.from_numpy(y)

class Pad1D:
    """
    Collate function: pads a batch of [T_i, D] to [B, T_max, D] with zeros.
    """
    def __call__(self, batch):
        xs, ys = zip(*batch)
        lens = [x.shape[0] for x in xs]
        Tm = max(lens); D = xs[0].shape[1]
        out = torch.zeros((len(xs), Tm, D), dtype=torch.float32)
        for i, x in enumerate(xs):
            t = x.shape[0]
            out[i, :t] = x
        ys = torch.stack(list(ys)).squeeze(-1)  # [B]
        return out, ys
