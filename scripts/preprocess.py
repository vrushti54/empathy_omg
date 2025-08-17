import os
import argparse
import numpy as np
import librosa
import soundfile as sf

def melspec(wav_path, sr=16000, n_mels=128, win_ms=25, hop_ms=10):
    y, orig_sr = sf.read(wav_path, dtype="float32")
    if y.ndim == 2:
        y = y.mean(axis=1)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        n_fft=int(sr*win_ms/1000),
        hop_length=int(sr*hop_ms/1000),
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    mu, sigma = S_db.mean(), S_db.std() + 1e-8
    S_db = (S_db - mu) / sigma
    return S_db.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    wavs = [f for f in os.listdir(args.wav_dir) if f.lower().endswith(".wav")]

    for i, f in enumerate(sorted(wavs), 1):
        clip_id = os.path.splitext(f)[0]
        in_path = os.path.join(args.wav_dir, f)
        out_path = os.path.join(args.out_dir, f"{clip_id}.npy")

        if os.path.exists(out_path):
            print(f"[{i}/{len(wavs)}] skip {clip_id} (exists)")
            continue

        try:
            M = melspec(in_path)
            np.save(out_path, M)
        except Exception as e:
            print(f"Error processing {clip_id}: {e}")

if __name__ == "__main__":
    main()
