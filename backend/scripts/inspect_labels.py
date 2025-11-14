import pandas as pd
import os
from collections import Counter

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
print(f"Scanning CSV files in: {os.path.abspath(DATA_DIR)}")
for fn in sorted(os.listdir(DATA_DIR)):
    if not fn.lower().endswith('.csv'):
        continue
    path = os.path.join(DATA_DIR, fn)
    try:
        df = pd.read_csv(path, nrows=2000)
    except Exception as e:
        print(f"{fn}: ERROR reading ({e})")
        continue
    # candidate label columns
    candidates = [c for c in df.columns if any(x in c.lower() for x in ['label','class','is_phish','is_phishing','phish','target','y','labelled'])]
    if not candidates:
        print(f"{fn}: no obvious label column found; columns (first 12): {list(df.columns)[:12]}")
        continue
    print(f"{fn}:")
    for c in candidates:
        sample_vals = df[c].dropna().unique().tolist()[:12]
        counts = Counter(df[c].dropna().astype(str).tolist())
        top = counts.most_common(10)
        print(f"  - detected label column '{c}' — unique sample: {sample_vals} ; top counts: {top}")
print('Done')
