import os, numpy as np, soundfile as sf, random
os.makedirs("data/raw/audio", exist_ok=True)
sr=16000; dur=2.0; t=np.linspace(0,dur,int(sr*dur),endpoint=False)
random.seed(42); np.random.seed(42)
N=60
ids=[]
for i in range(N):
    cid=f"d{i:03d}"
    f0 = random.choice([110,220,330,440,550])
    y = 0.15*np.sin(2*np.pi*f0*t) + 0.05*np.random.randn(t.size)
    sf.write(f"data/raw/audio/{cid}.wav", y.astype("float32"), sr)
    ids.append(cid)

# labels in [-1,1] with a weak correlation to f0 choice
vals = np.linspace(-1,1,N) + 0.1*np.random.randn(N)
np.random.shuffle(vals)
with open("data/labels/labels.csv","w") as f:
    f.write("id,target\n")
    for cid,v in zip(ids, vals): f.write(f"{cid},{float(v)}\n")
print("Wrote", N, "dummy clips and labels")
