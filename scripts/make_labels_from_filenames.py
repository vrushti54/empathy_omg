import os, csv, glob
ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob("data/raw/audio/*.wav"))
os.makedirs("data/labels", exist_ok=True)
with open("data/labels/labels.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["id","target"])
    # made-up targets; replace with real OMG labels later
    for i, cid in enumerate(ids):
        w.writerow([cid, (-1 + 1.0*i/max(1,len(ids)-1))])  # spread in [-1,1]
print("labels.csv ready with", len(ids), "rows")
