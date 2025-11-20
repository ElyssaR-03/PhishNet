"""Dataset validator for PhishNet CSVs.

Usage:
  python scripts/validate_dataset.py            # validate all CSVs under backend/data
  python scripts/validate_dataset.py file.csv  # validate a specific CSV path

Checks performed:
- CSV is readable (pandas with python engine, falls back to csv module)
- A label column exists and can be mapped to binary 0/1
- Reports class balance and missing-value rates
- Reports numeric/non-numeric feature counts

Exits with code 0 when all files are OK, non-zero when issues are found.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import warnings
from typing import Dict, List, Optional, Tuple

import pandas as pd


LABEL_CANDIDATES = ['is_phishing', 'class_label', 'CLASS_LABEL', 'label', 'target', 'is_phish', 'phishing']
SUBSTRING_TOKENS = ['label', 'class', 'phish', 'target', 'malicious']


def read_csv_tolerant(path: str) -> Optional[pd.DataFrame]:
    """Try reading CSV with pandas (python engine); fallback to csv module if necessary."""
    try:
        return pd.read_csv(path, engine='python', dtype=str)
    except Exception:
        try:
            with open(path, 'r', encoding='utf-8', newline='') as fh:
                reader = csv.reader(fh)
                rows = list(reader)
            if not rows:
                return None
            header = rows[0]
            data = rows[1:]
            return pd.DataFrame(data, columns=header)
        except Exception:
            return None


def detect_label_column(df: pd.DataFrame, explicit: Optional[str] = None) -> Optional[str]:
    cols = list(df.columns)
    if explicit:
        if explicit in cols:
            return explicit
        lmap = {c.lower(): c for c in cols}
        if explicit.lower() in lmap:
            return lmap[explicit.lower()]
        return None

    for cand in LABEL_CANDIDATES:
        if cand in cols:
            return cand
    cols_lower = {c.lower(): c for c in cols}
    for cand in LABEL_CANDIDATES:
        if cand in cols_lower:
            return cols_lower[cand]
    for c in cols:
        cl = c.lower()
        for token in SUBSTRING_TOKENS:
            if token in cl:
                return c
    return None


def map_label_value(v: str) -> Optional[int]:
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    pos = {'phishing', 'phish', 'malicious', '1', 'true', 'yes', 'y'}
    neg = {'legitimate', 'benign', 'legit', '0', 'false', 'no', 'n'}
    if s in pos:
        return 1
    if s in neg:
        return 0
    try:
        n = float(s)
        return 1 if n != 0 else 0
    except Exception:
        return None


def analyze_file(path: str) -> Tuple[bool, Dict]:
    """Validate a single CSV. Returns (ok, report)."""
    report: Dict = {'path': path}
    df = read_csv_tolerant(path)
    if df is None:
        report['error'] = 'Failed to read CSV'
        return False, report

    report['n_rows'] = len(df)
    report['n_columns'] = len(df.columns)
    label_col = detect_label_column(df)
    report['label_column'] = label_col
    if not label_col:
        report['error'] = 'No label column detected'
        return False, report

    # Inspect label mapping on a sample (or full column if small)
    col = df[label_col].astype(str)
    sample = col.dropna() if len(col) <= 1000 else col.dropna().sample(n=1000, random_state=0)
    mapped = [map_label_value(v) for v in sample]
    mapped_counts = {str(k): mapped.count(k) for k in (0, 1, None)}
    report['label_sample_counts'] = mapped_counts

    if mapped_counts.get('None', 0) > 0:
        # If many unmapped values, warn and fail
        none_frac = mapped_counts.get('None', 0) / max(1, sum(mapped_counts.values()))
        report['warning'] = f'{mapped_counts.get("None",0)} unmapped label values (fraction {none_frac:.2f})'
        if none_frac > 0.05:
            report['error'] = 'Too many unmapped label values (>5%)'
            return False, report

    # Convert label column via mapping and compute class balance
    mapped_full = [map_label_value(v) for v in df[label_col].astype(str).fillna('')]
    counts = {0: 0, 1: 0, 'unknown': 0}
    for mv in mapped_full:
        if mv is None:
            counts['unknown'] += 1
        else:
            counts[mv] += 1
    report['label_counts'] = counts
    if counts['unknown'] > 0:
        unk_frac = counts['unknown'] / max(1, len(df))
        report['error'] = f'{counts["unknown"]} unknown label values (fraction {unk_frac:.3f})'
        return False, report

    # Missing value report
    missing = df.isna().mean().to_dict()
    # convert float to percent rounded
    report['missing_fraction'] = {k: float(v) for k, v in missing.items()}

    # Numeric-features check: how many columns are numeric-like
    numeric_stats = {}
    numeric_count = 0
    for c in df.columns:
        if c == label_col:
            continue
        series = pd.to_numeric(df[c], errors='coerce')
        numeric_prop = series.notna().sum() / max(1, len(series))
        numeric_stats[c] = float(numeric_prop)
        if numeric_prop >= 0.9:
            numeric_count += 1
    report['numeric_feature_fraction'] = numeric_count / max(1, (len(df.columns) - 1))
    report['numeric_column_props'] = {k: v for k, v in numeric_stats.items() if v < 1.0}

    return True, report


def find_csvs(data_dir: str) -> List[str]:
    if not os.path.exists(data_dir):
        return []
    files = []
    for f in os.listdir(data_dir):
        if f.lower().endswith('.csv'):
            files.append(os.path.join(data_dir, f))
    return files


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Validate dataset CSVs for PhishNet')
    p.add_argument('paths', nargs='*', help='CSV file(s) to validate (optional)')
    p.add_argument('--data-dir', default=None, help='Directory to search for CSVs (defaults to backend/data)')
    args = p.parse_args(argv)

    base_dir = os.path.dirname(os.path.dirname(__file__))
    default_data_dir = args.data_dir if args.data_dir else os.path.join(base_dir, 'data')

    targets = []
    if args.paths:
        for pth in args.paths:
            targets.append(pth if os.path.isabs(pth) else os.path.join(default_data_dir, pth))
    else:
        targets = find_csvs(default_data_dir)

    if not targets:
        print(f'No CSV files found in {default_data_dir}', file=sys.stderr)
        return 2

    all_ok = True
    reports = []
    for f in targets:
        ok, rpt = analyze_file(f)
        reports.append(rpt)
        print(json.dumps(rpt, indent=2))
        if not ok:
            all_ok = False

    if all_ok:
        print('\nAll dataset checks passed')
        return 0
    else:
        print('\nOne or more dataset checks failed', file=sys.stderr)
        return 3


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
