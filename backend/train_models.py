"""
Training script for PhishNet ML models.

This script discovers CSV files under `backend/data/`, detects a label
column, normalizes labels to `is_phishing` (0/1), concatenates datasets,
and trains the PhishingDetector on the combined data. It persists the
trained models and the feature name ordering to the models directory.
"""
import os
import sys
import warnings
import json
from typing import List

import pandas as pd

from models.ml_models import PhishingDetector


def _normalize_label_series(s: pd.Series) -> pd.Series:
    # Reuse a robust normalization approach: numeric 0/1 preserved, common strings mapped
    vals = s.dropna().unique()
    if pd.api.types.is_numeric_dtype(s):
        uvals = set(pd.Series(vals).tolist())
        if uvals <= {0, 1}:
            return s.fillna(0).astype(int)
        if len(uvals) == 2:
            mn = min(uvals)
            return s.fillna(mn).apply(lambda v: 1 if v != mn else 0).astype(int)
        return s.fillna(0).apply(lambda v: 1 if v and v != 0 else 0).astype(int)

    mapping_pos = {'phishing', 'phish', 'malicious', '1', 'true', 'yes', 'y'}
    mapping_neg = {'legitimate', 'benign', 'legit', '0', 'false', 'no', 'n'}

    def str_to_label(v):
        if pd.isna(v):
            return 0
        vs = str(v).strip().lower()
        if vs in mapping_pos:
            return 1
        if vs in mapping_neg:
            return 0
        try:
            vn = float(vs)
            return 1 if vn != 0 else 0
        except Exception:
            return None

    converted = s.map(str_to_label)
    if converted.notna().all():
        return converted.astype(int)

    uniq = list(pd.Series(vals).astype(str))
    if len(uniq) == 2:
        a, b = uniq[0], uniq[1]
        warnings.warn(f"Unrecognized label strings {uniq}; mapping '{a}'->0 and '{b}'->1")
        return s.fillna(a).astype(str).map({a: 0, b: 1}).astype(int)

    raise ValueError("Could not normalize label column to binary 0/1")


def train_phishnet_models(data_dir: str = None, data_file: str = None, label_column: str = None, n_samples: int = 0):
    """Train PhishNet models using CSV files.

    Parameters
    - data_dir: optional directory to search for CSVs (defaults to backend/data)
    - data_file: optional specific CSV file to use
    - label_column: optional explicit label column name
    - n_samples: if >0 and no CSVs are found, generate a synthetic dataset of this size (controlled fallback)
    """
    print("PhishNet Model Training")
    print("=" * 50)

    default_data_dir = data_dir if data_dir else os.path.join(os.path.dirname(__file__), 'data')

    csv_files: List[str] = []
    if data_file:
        candidate = data_file if os.path.isabs(data_file) else os.path.join(default_data_dir, data_file)
        if os.path.exists(candidate) and candidate.lower().endswith('.csv'):
            csv_files = [candidate]
        else:
            raise FileNotFoundError(f"Specified data file not found: {candidate}")
    else:
        try:
            for fname in os.listdir(default_data_dir):
                if fname.lower().endswith('.csv'):
                    csv_files.append(os.path.join(default_data_dir, fname))
        except Exception:
            csv_files = []

    label_candidates = ['is_phishing', 'class_label', 'CLASS_LABEL', 'label', 'target', 'is_phish', 'phishing']
    substring_tokens = ['label', 'class', 'phish', 'target', 'malicious']

    datasets = []
    for path in csv_files:
        try:
            # Use the Python engine and read as strings to avoid C-engine tokenization
            # issues on poorly quoted CSVs. We'll coerce numeric features later.
            df = pd.read_csv(path, engine='python', dtype=str)
        except Exception as e:
            # Fallback: try Python's csv module which can be more tolerant in some cases
            import csv
            try:
                with open(path, 'r', encoding='utf-8', newline='') as fh:
                    reader = csv.reader(fh)
                    rows = list(reader)
                if not rows:
                    raise ValueError('No rows parsed')
                header = rows[0]
                data = rows[1:]
                df = pd.DataFrame(data, columns=header)
            except Exception:
                warnings.warn(f"Failed to read {path}: {e}")
                continue

        # determine label column
        found_label = None
        if label_column:
            if label_column in df.columns:
                found_label = label_column
            elif label_column.lower() in {c.lower(): c for c in df.columns}:
                found_label = {c.lower(): c for c in df.columns}[label_column.lower()]
            else:
                raise ValueError(f"Label column '{label_column}' not found in file {path}; available columns: {list(df.columns)[:20]}")
        else:
            for cand in label_candidates:
                if cand in df.columns:
                    found_label = cand
                    break
            if not found_label:
                cols_lower = {c.lower(): c for c in df.columns}
                for cand in label_candidates:
                    if cand in cols_lower:
                        found_label = cols_lower[cand]
                        break
            if not found_label:
                for col in df.columns:
                    col_l = col.lower()
                    for token in substring_tokens:
                        if token in col_l:
                            found_label = col
                            break
                    if found_label:
                        break

        if not found_label:
            warnings.warn(f"Skipping {path}: missing label column (tried {label_candidates})")
            continue

        try:
            df['is_phishing'] = _normalize_label_series(df[found_label])
            print(f"Using label column '{found_label}' from file '{os.path.basename(path)}' -> normalized to 'is_phishing'")
            if found_label != 'is_phishing':
                try:
                    df = df.drop(columns=[found_label])
                except Exception:
                    pass
            try:
                counts = df['is_phishing'].value_counts().to_dict()
            except Exception:
                counts = {}
            print(f"Including dataset: {path} ({len(df)} rows) [label='{found_label}'] - label counts: {counts}")
            datasets.append(df)
        except Exception as e:
            warnings.warn(f"Skipping {path}: could not normalize label column '{found_label}': {e}")

    if datasets:
        combined = pd.concat(datasets, ignore_index=True)
        expected_features = [c for c in combined.columns if c != 'is_phishing']
        X_combined = pd.DataFrame(index=combined.index)
        for feat in expected_features:
            X_combined[feat] = pd.to_numeric(combined[feat], errors='coerce').fillna(0)
        y = combined['is_phishing'].values
        X = X_combined
        print(f"Loaded combined dataset with {len(X)} samples from {len(datasets)} file(s)")
    else:
        available = ', '.join(os.listdir(default_data_dir)) if os.path.exists(default_data_dir) else 'none'
        raise FileNotFoundError(
            f"No CSV datasets with a detectable label column were found in {default_data_dir}. Found files: {available}.\nPlease add your dataset CSV (e.g. 'Phishing_Legitimate_full.csv')"
        )

    print(f"\nDataset size: {len(X)} samples")
    print(f"Legitimate samples: {sum(y == 0)}")
    print(f"Phishing samples: {sum(y == 1)}")
    print(f"Features: {list(X.columns)}")

    # Initialize detector
    print("\nInitializing PhishingDetector...")
    detector = PhishingDetector()
    # Record feature names on the detector so saved models know the training order
    try:
        detector.feature_names = list(X.columns)
    except Exception:
        detector.feature_names = None

    # Train models
    print("\nTraining models...")
    scores = detector.train(X.values, y)

    print("\nTraining completed!")
    print("\nModel Accuracies:")
    print("-" * 50)
    for model_name, accuracy in scores.items():
        print(f"{model_name.replace('_', ' ').title()}: {accuracy:.4f} ({accuracy*100:.2f}%)")

    # Persist feature list alongside models so predictions can be aligned later
    features_path = os.path.join(detector.model_dir, 'feature_names.json')
    try:
        with open(features_path, 'w', encoding='utf-8') as fh:
            json.dump(list(X.columns), fh)
        print(f"Saved feature list to {features_path}")
    except Exception as e:
        warnings.warn(f"Failed to save feature list to {features_path}: {e}")

    # Save models
    print("\nSaving trained models...")
    detector.save_models()
    print("Models saved successfully!")

    return detector, scores


if __name__ == '__main__':
    args = None
    try:
        detector, scores = train_phishnet_models()
        print("\n" + "=" * 50)
        print("Training completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        sys.exit(1)
    finally:
        # Restore os.listdir if we overrode it
        try:
            if 'original_listdir' in locals():
                os.listdir = original_listdir
        except Exception:
            pass
