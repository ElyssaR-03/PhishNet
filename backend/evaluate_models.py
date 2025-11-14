"""
Evaluate trained models on combined datasets and print per-class metrics.
"""
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from feature_extractor import FeatureExtractor
from models.ml_models import PhishingDetector


def load_combined_dataset(data_dir=None):
    if data_dir is None:
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
    csv_files = []
    for fname in os.listdir(data_dir):
        if fname.lower().endswith('.csv'):
            csv_files.append(os.path.join(data_dir, fname))

    label_candidates = [
        'is_phishing', 'class_label', 'classlabel', 'is_phish', 'phishing', 'label', 'target', 'is_malicious'
    ]
    substring_tokens = ['label', 'class', 'phish', 'target', 'malicious']

    def _normalize_label_series(s: pd.Series) -> pd.Series:
        vals = s.dropna().unique()
        if pd.api.types.is_numeric_dtype(s):
            uvals = set(vals.tolist())
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

    datasets = []
    for path in csv_files:
        try:
            df = pd.read_csv(path)
        except Exception as e:
            warnings.warn(f"Failed to read {path}: {e}")
            continue

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
            if found_label != 'is_phishing':
                try:
                    df = df.drop(columns=[found_label])
                except Exception:
                    pass
            datasets.append(df)
            print(f"Included {os.path.basename(path)} as label='{found_label}' ({len(df)} rows)")
        except Exception as e:
            warnings.warn(f"Skipping {path}: could not normalize label column '{found_label}': {e}")

    if not datasets:
        raise RuntimeError("No datasets found with a usable label column")

    combined = pd.concat(datasets, ignore_index=True)
    expected_features = FeatureExtractor.get_feature_names()
    X = pd.DataFrame()
    for feat in expected_features:
        if feat in combined.columns:
            X[feat] = pd.to_numeric(combined[feat], errors='coerce').fillna(0)
        else:
            X[feat] = 0
    y = combined['is_phishing'].values.astype(int)
    return X, y


if __name__ == '__main__':
    X, y = load_combined_dataset()
    print(f"Total samples: {len(X)}; Positive (phishing): {sum(y==1)}; Negative: {sum(y==0)}")

    # For quick evaluation, sample a stratified subset if dataset is large
    max_samples = 2000
    if len(X) > max_samples:
        X_sub, _, y_sub, _ = train_test_split(X.values, y, train_size=max_samples, stratify=y, random_state=42)
        X_use = X_sub
        y_use = y_sub
    else:
        X_use = X.values
        y_use = y

    X_train, X_test, y_train, y_test = train_test_split(X_use, y_use, test_size=0.2, random_state=42, stratify=y_use)

    detector = PhishingDetector()
    # train models on training split (need to scale and fit)
    detector.scaler.fit(X_train)
    X_train_scaled = detector.scaler.transform(X_train)
    X_test_scaled = detector.scaler.transform(X_test)

    for name, model in detector.models.items():
        print('\n' + '='*60)
        print(f"Training and evaluating model: {name}")
        # clone fresh estimator instance
        # fit
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        print("Accuracy:", np.mean(preds == y_test))
        print("Classification report:\n", classification_report(y_test, preds, digits=4))
        print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Optionally save models
    detector.is_trained = True
    detector.save_models()
    print('\nSaved trained models to', detector.model_dir)
