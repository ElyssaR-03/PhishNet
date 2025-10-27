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
