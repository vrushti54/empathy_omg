import argparse, os, json, pandas as pd

def write_split(items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f: json.dump(items, f, indent=2)

def from_one_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = {"id","target","split"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{csv_path} must contain columns: {required}")
    for name in ["train","val","test"]:
        part = df[df["split"].str.lower()==name][["id","target"]]
        items = [{"id": str(r.id), "target": float(r.target)} for r in part.itertuples(index=False)]
        write_split(items, f"data/splits/{name}.json")
        print(name, len(items))

def from_three_csvs(csv_dir):
    for name in ["train","val","test"]:
        p = os.path.join(csv_dir, f"{name}.csv")
        if not os.path.exists(p): raise SystemExit(f"Missing {p}")
        df = pd.read_csv(p)
        if not {"id","target"}.issubset(df.columns):
            raise SystemExit(f"{p} must contain columns: id,target")
        items = [{"id": str(r.id), "target": float(r.target)} for r in df.itertuples(index=False)]
        write_split(items, f"data/splits/{name}.json")
        print(name, len(items))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--one_csv", help="Single CSV with columns id,target,split")
    ap.add_argument("--dir_csv", help="Directory with train.csv/val.csv/test.csv (id,target)")
    args = ap.parse_args()
    if bool(args.one_csv) == bool(args.dir_csv):
        raise SystemExit("Provide exactly one of --one_csv or --dir_csv")
    if args.one_csv: from_one_csv(args.one_csv)
    else: from_three_csvs(args.dir_csv)

if __name__ == "__main__":
    main()
