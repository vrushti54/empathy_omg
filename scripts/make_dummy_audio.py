import os, numpy as np, soundfile as sf
os.makedirs("data/raw/audio", exist_ok=True)
sr = 16000
dur = 2.0
t = np.linspace(0, dur, int(sr*dur), endpoint=False)
clips = {
    "a001": 0.2*np.sin(2*np.pi*220*t),          # 220 Hz tone
    "a002": 0.2*np.sin(2*np.pi*440*t),          # 440 Hz tone
    "a003": 0.2*np.random.randn(t.size),        # noise
}
for name, y in clips.items():
    sf.write(f"data/raw/audio/{name}.wav", y.astype("float32"), sr)
print("Wrote", len(clips), "dummy wavs at data/raw/audio")
