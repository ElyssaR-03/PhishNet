"""
Training script for PhishNet ML models.
"""
import os
import sys
import argparse
import warnings
import pandas as pd
import json
# Use package-style imports so this module works when invoked as a package
from backend.feature_extractor import FeatureExtractor
from backend.models.ml_models import PhishingDetector


def train_phishnet_models(data_dir: str = None, data_file: str = None, label_column: str = None):
    """Train all PhishNet models on the dataset.

    This function now looks for CSV files under `backend/data/` and will
    concatenate all files that contain an `is_phishing` column. If none are
    found, it falls back to the existing single-file behavior and synthetic
    generation.
    """
    print("PhishNet Model Training")
    print("=" * 50)

    # Default data directory (can be overridden via CLI)
    default_data_dir = data_dir if data_dir else os.path.join(os.path.dirname(__file__), 'data')

    # Discover CSV files in data directory or use single file if provided
    csv_files = []
    if data_file:
        # If an absolute path was provided use it; otherwise join with default_data_dir
        if os.path.isabs(data_file):
            candidate = data_file
        else:
            candidate = os.path.join(default_data_dir, data_file)
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

    # Load and concatenate CSVs, accepting multiple label column names and normalizing
    # Exact candidates and safer substring tokens (avoid single-letter tokens like 'y')
    label_candidates = [
        'is_phishing', 'class_label', 'classlabel', 'is_phish', 'phishing', 'label', 'target', 'is_malicious'
    ]
    substring_tokens = ['label', 'class', 'phish', 'target', 'malicious']

    def _normalize_label_series(s: pd.Series) -> pd.Series:
        """Normalize a pandas Series to 0/1 values for phishing label.

        Heuristics:
        - If numeric and values are subset of {0,1} -> keep.
        - If numeric with two unique values -> map min->0, max->1.
        - If strings, map common phishing tokens to 1 and benign tokens to 0.
        - If two unique strings not recognized, map first->0 second->1 (with warning).
        """
        # Drop NA for decision making but preserve indices
        vals = s.dropna().unique()

        # Numeric pathway
        if pd.api.types.is_numeric_dtype(s):
            uvals = set(vals.tolist())
            if uvals <= {0, 1}:
                return s.fillna(0).astype(int)
            if len(uvals) == 2:
                mn = min(uvals)
                return s.fillna(mn).apply(lambda v: 1 if v != mn else 0).astype(int)
            # Fallback: any positive -> phishing
            return s.fillna(0).apply(lambda v: 1 if v and v != 0 else 0).astype(int)

        # String/object pathway: map common tokens
        mapping_pos = {'phishing', 'phish', 'malicious', 'phishng', 'phishng', '1', 'true', 'yes', 'y', 'phish/true'}
        mapping_neg = {'legitimate', 'benign', 'legit', '0', 'false', 'no', 'n'}

        def str_to_label(v):
            if pd.isna(v):
                return 0
            vs = str(v).strip().lower()
            if vs in mapping_pos:
                return 1
            if vs in mapping_neg:
                return 0
            # try numeric string
            try:
                vn = float(vs)
                return 1 if vn != 0 else 0
            except Exception:
                return None

        converted = s.map(str_to_label)
        if converted.notna().all():
            return converted.astype(int)

        # If some values couldn't be recognized but there are exactly two unique original values,
        # map them deterministically (first->0, second->1)
        uniq = list(pd.Series(vals).astype(str))
        if len(uniq) == 2:
            a, b = uniq[0], uniq[1]
            warnings.warn(f"Unrecognized label strings {uniq}; mapping '{a}'->0 and '{b}'->1")
            return s.fillna(a).astype(str).map({a: 0, b: 1}).astype(int)

        raise ValueError("Could not normalize label column to binary 0/1")

    datasets = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")
            continue

        # determine label column
        if label_column:
            # user explicitly requested a label column name
            if label_column in df.columns:
                found_label = label_column
            elif label_column.lower() in {c.lower(): c for c in df.columns}:
                # case-insensitive match
                found_label = {c.lower(): c for c in df.columns}[label_column.lower()]
            else:
                raise ValueError(f"Label column '{label_column}' not found in file {path}; available columns: {list(df.columns)[:20]}")
        else:
            # find label column (exact case-sensitive then case-insensitive)
            found_label = None
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

            # If still not found, try safer substring matching using longer tokens
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

        # Normalize label
        try:
            df['is_phishing'] = _normalize_label_series(df[found_label])
            print(f"Using label column '{found_label}' from file '{os.path.basename(path)}' -> normalized to 'is_phishing'")
            # Drop the original label column if it wasn't already 'is_phishing'
            if found_label != 'is_phishing':
                try:
                    df = df.drop(columns=[found_label])
                except Exception:
                    pass
            # Print label distribution for transparency
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
        # Use the feature columns present in the CSV(s) (excluding the label) as the training feature set.
        # This allows training to use the exact features provided in the dataset file(s).
        expected_features = [c for c in combined.columns if c != 'is_phishing']

        # Create DataFrame indexed like the combined data.
        X_combined = pd.DataFrame(index=combined.index)
        for feat in expected_features:
            # Coerce to numeric where possible and fill missing values with 0
            X_combined[feat] = pd.to_numeric(combined[feat], errors='coerce').fillna(0)

        # Allow additional features to be ignored (models expect fixed feature order)
        y = combined['is_phishing'].values
        X = X_combined
        print(f"Loaded combined dataset with {len(X)} samples from {len(datasets)} file(s)")
    else:
        # Do not generate synthetic data automatically. Require an explicit dataset file.
        available = ', '.join(os.listdir(default_data_dir)) if os.path.exists(default_data_dir) else 'none'
        raise FileNotFoundError(
            f"No CSV datasets with a detectable label column were found in {default_data_dir}."
            f" Found files: {available}.\nPlease add your dataset CSV (e.g. 'Phishing_Legitimate_full.csv')"
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train PhishNet models")
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Optional directory containing CSV dataset files (overrides default)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # If user provided a data-dir, temporarily adjust the data directory search
    if args.data_dir:
        # Monkey-patch default data dir by changing environment variable used by this script
        # Simpler: change current working directory when searching; pass in via global variable would be heavier
        # We'll set the expected data directory path by replacing the folder used inside the function.
        # Easiest approach: change os.path.dirname(__file__) reference by chdir; do it carefully.
        try:
            os.chdir(os.path.dirname(__file__))
        except Exception:
            pass
        # If provided path is absolute, use it; otherwise join to script dir
        if os.path.isabs(args.data_dir):
            data_dir = args.data_dir
        else:
            data_dir = os.path.join(os.path.dirname(__file__), args.data_dir)
        # Temporarily rename default_data_dir variable by injecting CSVs into expected folder path
        # We'll create a small wrapper to call train_phishnet_models while temporarily pointing os.listdir
        original_listdir = os.listdir
        def _listdir_override(path):
            if path == os.path.join(os.path.dirname(__file__), 'data'):
                try:
                    return original_listdir(data_dir)
                except Exception:
                    return []
            return original_listdir(path)
        os.listdir = _listdir_override

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
