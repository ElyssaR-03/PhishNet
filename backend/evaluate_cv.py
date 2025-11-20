"""
Run stratified k-fold cross-validation for each model and print averaged metrics.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from feature_extractor import FeatureExtractor
from models.ml_models import PhishingDetector
from evaluate_models import load_combined_dataset


def evaluate_cv(n_splits=5, max_samples=2000, random_state=42):
    X, y = load_combined_dataset()
    # downsample if necessary
    if len(X) > max_samples:
        from sklearn.model_selection import train_test_split
        X_sub, _, y_sub, _ = train_test_split(X.values, y, train_size=max_samples, stratify=y, random_state=random_state)
        X = pd.DataFrame(X_sub, columns=FeatureExtractor.get_feature_names())
        y = y_sub
    else:
        X = X
        y = y

    X = X.values.astype(float)
    y = np.array(y).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    detector = PhishingDetector()
    results = {}

    for name, base_model in detector.models.items():
        print('\n' + '='*60)
        print(f"Cross-validating model: {name}")
        y_true_all = []
        y_pred_all = []
        y_score_all = []

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # fresh model instance
            # sklearn estimators are stateful; create new by cloning from base if possible
            from sklearn.base import clone
            model = clone(base_model)

            # scaler per-fold
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(X_train)
            X_train_s = scaler.transform(X_train)
            X_test_s = scaler.transform(X_test)

            # train
            model.fit(X_train_s, y_train)

            # predict
            y_pred = model.predict(X_test_s)
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())

            # score/probability if available
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test_s)[:, 1]
            else:
                # use decision function if available and scaled to [0,1]
                if hasattr(model, 'decision_function'):
                    df = model.decision_function(X_test_s)
                    # min-max scale
                    if np.ptp(df) == 0:
                        proba = np.zeros_like(df)
                    else:
                        proba = (df - df.min()) / (df.max() - df.min())
                else:
                    proba = np.zeros_like(y_test)
            y_score_all.extend(proba.tolist())

        # compute metrics
        report = classification_report(y_true_all, y_pred_all, digits=4, output_dict=True)
        try:
            auc = roc_auc_score(y_true_all, y_score_all)
        except Exception:
            auc = None

        results[name] = {'report': report, 'auc': auc}

        # Print concise summary
        print(f"ROC AUC: {auc:.4f}" if auc is not None else "ROC AUC: n/a")
        print("Classification report (macro avg):")
        print({
            'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'],
            'f1-score': report['macro avg']['f1-score']
        })
        print('\nFull report:')
        print(classification_report(y_true_all, y_pred_all, digits=4))

    return results


if __name__ == '__main__':
    res = evaluate_cv()
    print('\nCross-validation completed')
