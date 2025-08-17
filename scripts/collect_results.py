import json, glob, os, csv

os.makedirs("results", exist_ok=True)
rows=[]
for path in sorted(glob.glob("results/*_audio_melspec.json")):
    name = os.path.splitext(os.path.basename(path))[0]  # e.g., mlp_audio_melspec
    with open(path) as f:
        m = json.load(f)
    rows.append({
        "model": name.replace("_audio_melspec","").upper(),
        "CCC": m.get("CCC"),
        "PCC": m.get("PCC"),
        "MAE": m.get("MAE"),
        "MSE": m.get("MSE"),
    })

# write CSV
csv_path = "results/summary.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["model","CCC","PCC","MAE","MSE"])
    w.writeheader(); w.writerows(rows)

# write Markdown
md_path = "results/summary.md"
with open(md_path, "w") as f:
    f.write("| Model | CCC | PCC | MAE | MSE |\n|---|---:|---:|---:|---:|\n")
    for r in rows:
        f.write(f"| {r['model']} | {r['CCC']:.4f} | {r['PCC']:.4f} | {r['MAE']:.4f} | {r['MSE']:.4f} |\n")

print("Wrote", csv_path, "and", md_path)
