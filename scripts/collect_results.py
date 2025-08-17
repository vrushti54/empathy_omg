import json, glob, csv, os

def load_json(p):
    try:
        with open(p) as f: return json.load(f)
    except FileNotFoundError:
        return None

rows=[]
# Models we track
models = ["mlp","lstm","gru","cnn1d"]

# 1) Validation rows (existing)
for m in models:
    p = f"results/{m}_audio_melspec.json"
    d = load_json(p)
    if d is not None:
        rows.append(dict(split="VAL", model=m.upper(), CCC=d["CCC"], PCC=d["PCC"], MAE=d["MAE"], MSE=d["MSE"]))

# 2) Test rows (new)
for m in models:
    p = f"results/test_{m}_audio_melspec.json"
    d = load_json(p)
    if d is not None:
        rows.append(dict(split="TEST", model=m.upper(), CCC=d["CCC"], PCC=d["PCC"], MAE=d["MAE"], MSE=d["MSE"]))

# Write CSV
os.makedirs("results", exist_ok=True)
csv_path = "results/summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["split","model","CCC","PCC","MAE","MSE"])
    w.writeheader(); w.writerows(rows)

# Write Markdown
md_path = "results/summary.md"
with open(md_path, "w") as f:
    f.write("| Split | Model | CCC | PCC | MAE | MSE |\n|---|---|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r['split']} | {r['model']} | {r['CCC']:.4f} | {r['PCC']:.4f} | {r['MAE']:.4f} | {r['MSE']:.4f} |\n")

print("Wrote", csv_path, "and", md_path)
