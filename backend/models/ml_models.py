"""
Machine Learning models for phishing detection.
Implements SVM, Random Forest, and Logistic Regression classifiers.
"""
import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import pandas as pd

# Optional SHAP import for explainability
try:
    import shap
except Exception:
    shap = None

class PhishingDetector:
    """Main class for phishing detection using multiple ML models."""
    
    def __init__(self):
        """Initialize the phishing detector with three ML models."""
        self.models = {
            'svm': SVC(kernel='rbf', probability=True, random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'logistic_regression': LogisticRegression(max_iter=1000, random_state=42)
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.model_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
        os.makedirs(self.model_dir, exist_ok=True)
        # explainers cache (model_name -> shap.Explainer)
        self.explainers: Dict[str, object] = {}
    
    def train(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Dict[str, float]:
        """
        Train all models on the provided data.
        
        Args:
            X: Feature matrix
            y: Target labels (0 for legitimate, 1 for phishing)
            test_size: Proportion of data to use for testing
            
        Returns:
            Dictionary with accuracy scores for each model
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train each model and evaluate
        scores = {}
        for name, model in self.models.items():
            model.fit(X_train_scaled, y_train)
            score = model.score(X_test_scaled, y_test)
            scores[name] = float(score)
        
        self.is_trained = True
        return scores
    
    def predict(self, X: np.ndarray, model_name: str = 'random_forest') -> Tuple[int, float]:
        """
        Predict whether input is phishing or legitimate.
        
        Args:
            X: Feature vector
            model_name: Which model to use ('svm', 'random_forest', 'logistic_regression')
            
        Returns:
            Tuple of (prediction, confidence) where prediction is 0/1 and confidence is probability
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Reshape if single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get prediction and probability
        model = self.models[model_name]
        prediction = model.predict(X_scaled)[0]
        
        # Get probability/confidence
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(proba[int(prediction)])
        else:
            confidence = 0.5  # Default if probability not available
        
        return int(prediction), confidence
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[int, float, Dict[str, int]]:
        """
        Predict using ensemble voting from all models.
        
        Args:
            X: Feature vector
            
        Returns:
            Tuple of (prediction, average_confidence, individual_predictions)
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        # Reshape if single sample
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        confidences = []
        
        for name, model in self.models.items():
            pred = int(model.predict(X_scaled)[0])
            predictions[name] = pred
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_scaled)[0]
                confidences.append(float(proba[pred]))
        
        # Majority voting
        votes = sum(predictions.values())
        ensemble_prediction = 1 if votes >= 2 else 0
        
        # Average confidence
        avg_confidence = float(np.mean(confidences)) if confidences else 0.5
        
        return ensemble_prediction, avg_confidence, predictions

    def explain(self, X: np.ndarray, model_name: str = 'random_forest', top_k: int = 5) -> Dict:
        """Return SHAP-based explanation (top_k features) for a single sample.

        - X: 1D numpy array (raw, unscaled) of feature values in training order
        - model_name: which model to explain (currently supports 'random_forest' and 'logistic_regression')
        """
        if not self.is_trained:
            raise ValueError("Models must be trained/loaded before explaining")

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # Ensure feature vector matches scaler expected input size (pad/truncate)
        expected = getattr(self.scaler, 'n_features_in_', None)
        if expected is not None:
            cur = X.shape[1]
            if cur < expected:
                pad = np.zeros((X.shape[0], expected - cur), dtype=float)
                X = np.concatenate([X, pad], axis=1)
            elif cur > expected:
                X = X[:, :expected]

        # scale using stored scaler
        Xs = self.scaler.transform(X)

        result: Dict = {
            'model': model_name,
            'scaled_vector': Xs.tolist(),
            'top_features': []
        }

        # Ensure explainer exists for model
        expl = self.explainers.get(model_name)
        model = self.models.get(model_name)

        # Create explainers lazily if shap is available
        if shap is not None and expl is None and model is not None:
            try:
                if model_name == 'random_forest':
                    expl = shap.TreeExplainer(model)
                elif model_name == 'logistic_regression':
                    # Use scaler mean as simple background
                    if hasattr(self.scaler, 'mean_'):
                        bg = self.scaler.mean_.reshape(1, -1)
                        expl = shap.LinearExplainer(model, bg)
                    else:
                        expl = shap.LinearExplainer(model, np.zeros((1, Xs.shape[1])))
                # cache
                if expl is not None:
                    self.explainers[model_name] = expl
            except Exception:
                expl = None

        if expl is None:
            # Fallback for linear models: use coefficients
            if model_name == 'logistic_regression' and model is not None and hasattr(model, 'coef_'):
                coefs = model.coef_[0]
                contributions = (coefs * Xs[0]).tolist()
                feature_names = getattr(self, 'feature_names', [f'f{i}' for i in range(len(contributions))])
                feats = [{'feature': n, 'value': float(v), 'contribution': float(c), 'abs': abs(c)} for n, v, c in zip(feature_names, X[0].tolist(), contributions)]
                feats_sorted = sorted(feats, key=lambda x: x['abs'], reverse=True)[:top_k]
                result['top_features'] = feats_sorted
                return result
            else:
                # No explainer available
                result['warning'] = 'No SHAP explainer available for this model'
                return result

        # Compute SHAP values
        try:
            shap_vals = expl.shap_values(Xs)
            # shap_values may be list (per-class) or array
            if isinstance(shap_vals, list):
                # pick class 1 (phishing) if present
                if len(shap_vals) > 1:
                    sv = np.array(shap_vals[1])[0]
                else:
                    sv = np.array(shap_vals[0])[0]
            else:
                sv = np.array(shap_vals)[0]

            feature_names = getattr(self, 'feature_names', [f'f{i}' for i in range(len(sv))])
            feats = []
            for name, raw_val, contrib in zip(feature_names, X[0].tolist(), sv.tolist()):
                feats.append({'feature': name, 'value': float(raw_val), 'contribution': float(contrib), 'abs': abs(contrib)})
            feats_sorted = sorted(feats, key=lambda x: x['abs'], reverse=True)[:top_k]
            result['top_features'] = feats_sorted
            return result
        except Exception as e:
            result['error'] = str(e)
            return result
    
    def save_models(self, prefix: str = "model") -> None:
        """Save trained models and scaler to disk."""
        if not self.is_trained:
            raise ValueError("Models must be trained before saving")
        
        for name, model in self.models.items():
            filepath = os.path.join(self.model_dir, f"{prefix}_{name}.joblib")
            joblib.dump(model, filepath)
        
        scaler_path = os.path.join(self.model_dir, f"{prefix}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
    
    def load_models(self, prefix: str = "model") -> bool:
        """
        Load trained models and scaler from disk.
        
        Returns:
            True if models loaded successfully, False otherwise
        """
        try:
            for name in self.models.keys():
                filepath = os.path.join(self.model_dir, f"{prefix}_{name}.joblib")
                if os.path.exists(filepath):
                    self.models[name] = joblib.load(filepath)
            
            scaler_path = os.path.join(self.model_dir, f"{prefix}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            
            self.is_trained = True
            return True
        except Exception:
            return False
