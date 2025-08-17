import pandas as pd, json, os
from sklearn.model_selection import train_test_split

os.makedirs("data/splits", exist_ok=True)
df = pd.read_csv("data/labels/labels.csv")  # id,target
n = len(df)

if n == 0:
    raise SystemExit("No rows in data/labels/labels.csv")
elif n == 1:
    train = val = test = df.copy()
elif n == 2:
    train = df.iloc[[0]].copy()
    val   = df.iloc[[1]].copy()
    test  = df.iloc[[1]].copy()
elif n == 3:
    train = df.iloc[[0,1]].copy()
    val   = df.iloc[[2]].copy()
    test  = df.iloc[[2]].copy()
else:
    train, temp = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
    # guard tiny temp sizes
    if len(temp) < 2:
        val = test = temp.copy()
    else:
        val, test = train_test_split(temp, test_size=0.50, random_state=42, shuffle=True)

for name, part in [("train",train), ("val",val), ("test",test)]:
    items = [{"id": str(r.id), "target": float(r.target)} for r in part.itertuples(index=False)]
    with open(f"data/splits/{name}.json","w") as f:
        json.dump(items, f, indent=2)
    print(name, len(items))
