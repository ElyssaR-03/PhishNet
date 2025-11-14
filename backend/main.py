"""
FastAPI backend for PhishNet phishing detection application.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import numpy as np
from models.ml_models import PhishingDetector
from feature_extractor import FeatureExtractor
import os
import warnings


# Helper to build feature vectors aligned to trained feature order
def build_feature_vector(detector: PhishingDetector, features: Dict[str, float], default_order: List[str]) -> np.ndarray:
    """Return a numpy array feature vector aligned to detector.feature_names if available.

    - If detector.feature_names exists, use that order; missing features are filled with 0.
    - Otherwise, use the provided default_order.
    - If the resulting vector length doesn't match scaler.n_features_in_ (when available), pad or truncate.
    """
    # Decide which order to use
    order = detector.feature_names if getattr(detector, 'feature_names', None) else default_order

    vec = np.array([features.get(k, 0) for k in order], dtype=float)

    # If scaler expects a certain number of features, try to adapt (pad with zeros or truncate)
    try:
        expected = getattr(detector.scaler, 'n_features_in_', None)
        if expected is not None:
            if len(vec) < expected:
                # pad with zeros
                pad = np.zeros(expected - len(vec), dtype=float)
                vec = np.concatenate([vec, pad])
            elif len(vec) > expected:
                warnings.warn(f"Incoming feature vector has {len(vec)} features but scaler expects {expected}; truncating")
                vec = vec[:expected]
    except Exception:
        # If anything goes wrong, return the vector as-is
        pass

    return vec


# Initialize FastAPI app
app = FastAPI(
    title="PhishNet API",
    description="API for phishing detection using machine learning",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models and feature extractor
detector = PhishingDetector()
feature_extractor = FeatureExtractor()

# Try to load pre-trained models
models_loaded = detector.load_models()


# Pydantic models for request/response
class URLAnalysisRequest(BaseModel):
    url: str = Field(..., description="URL to analyze for phishing")
    model: str = Field("random_forest", description="ML model to use (svm, random_forest, logistic_regression, ensemble)")


class EmailAnalysisRequest(BaseModel):
    content: str = Field(..., description="Email content to analyze")
    sender: Optional[str] = Field("", description="Sender email address")
    model: str = Field("random_forest", description="ML model to use")


class AnalysisResponse(BaseModel):
    is_phishing: bool
    confidence: float
    risk_level: str
    features: Dict[str, float]
    model_used: str
    individual_predictions: Optional[Dict[str, int]] = None


class TrainingRequest(BaseModel):
    n_samples: int = Field(1000, description="Number of samples to generate for training")


class TrainingResponse(BaseModel):
    success: bool
    message: str
    accuracies: Dict[str, float]


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


# Helper functions
def determine_risk_level(confidence: float, is_phishing: bool) -> str:
    """Determine risk level based on prediction and confidence."""
    if not is_phishing:
        return "Safe"
    elif confidence >= 0.8:
        return "High Risk"
    elif confidence >= 0.6:
        return "Medium Risk"
    else:
        return "Low Risk"


# API endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return {
        "status": "PhishNet API is running",
        "models_loaded": detector.is_trained
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy" if detector.is_trained else "models not trained",
        "models_loaded": detector.is_trained
    }


@app.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url(request: URLAnalysisRequest):
    """
    Analyze a URL for phishing indicators.
    """
    if not detector.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Models not trained. Please train models first using /train endpoint."
        )
    
    try:
        # Extract features
        features = feature_extractor.extract_url_features(request.url)

        # Default URL feature order used when no trained feature list available
        default_url_order = [
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
            'num_questions', 'num_equals', 'num_at', 'num_ampersands', 'num_digits',
            'has_ip', 'is_https', 'domain_length', 'path_length', 'query_length',
            'has_suspicious_keywords'
        ]

        feature_vector = build_feature_vector(detector, features, default_url_order)
        
        # Make prediction
        if request.model == "ensemble":
            prediction, confidence, individual_preds = detector.predict_ensemble(feature_vector)
            model_used = "ensemble"
        else:
            prediction, confidence = detector.predict(feature_vector, request.model)
            individual_preds = None
            model_used = request.model
        
        is_phishing = bool(prediction == 1)
        risk_level = determine_risk_level(confidence, is_phishing)
        
        return AnalysisResponse(
            is_phishing=is_phishing,
            confidence=confidence,
            risk_level=risk_level,
            features=features,
            model_used=model_used,
            individual_predictions=individual_preds
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze/email", response_model=AnalysisResponse)
async def analyze_email(request: EmailAnalysisRequest):
    """
    Analyze an email for phishing indicators.
    """
    if not detector.is_trained:
        raise HTTPException(
            status_code=503,
            detail="Models not trained. Please train models first using /train endpoint."
        )
    
    try:
        # Extract features
        features = feature_extractor.extract_email_features(request.content, request.sender)
        
        # For email, we'll use a subset of features that are relevant.
        # Default email-to-URL-mapped order (padding included) used if no trained feature list exists.
        default_email_order = [
            'content_length', 'num_urls', 'num_suspicious_keywords', 'has_money_keywords',
            'num_exclamations', 'capital_ratio', 'mentions_attachments', 'sender_length',
            'sender_has_numbers',
            # padding entries to match URL feature space size
            'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes', 'num_questions', 'num_equals'
        ]

        # Normalize/scale some email features to be roughly comparable to URL features
        features['content_length'] = features.get('content_length', 0) / 10
        features['has_money_keywords'] = features.get('has_money_keywords', 0) * 10
        features['capital_ratio'] = features.get('capital_ratio', 0) * 100
        features['mentions_attachments'] = features.get('mentions_attachments', 0) * 10
        features['sender_has_numbers'] = features.get('sender_has_numbers', 0) * 10

        feature_vector = build_feature_vector(detector, features, default_email_order)
        
        # Make prediction
        if request.model == "ensemble":
            prediction, confidence, individual_preds = detector.predict_ensemble(feature_vector)
            model_used = "ensemble"
        else:
            prediction, confidence = detector.predict(feature_vector, request.model)
            individual_preds = None
            model_used = request.model
        
        is_phishing = bool(prediction == 1)
        risk_level = determine_risk_level(confidence, is_phishing)
        
        return AnalysisResponse(
            is_phishing=is_phishing,
            confidence=confidence,
            risk_level=risk_level,
            features=features,
            model_used=model_used,
            individual_predictions=individual_preds
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/train", response_model=TrainingResponse)
async def train_models(request: TrainingRequest):
    """
    Train the ML models using datasets present in the data directory (or the configured data file).
    """
    try:
        # Use the train_models utility which loads CSV(s) from backend/data/ (or accepts --data-file)
        # It returns a trained detector and scores.
        from train_models import train_phishnet_models

        # call training (uses files in backend/data by default). Ignore request.n_samples here.
        new_detector, scores = train_phishnet_models()

        # Replace module-level detector with the newly trained instance so API endpoints use it
        global detector
        detector = new_detector
        
        return TrainingResponse(
            success=True,
            message=f"Models trained successfully using datasets in backend/data/",
            accuracies=scores
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.get("/models/info")
async def get_models_info():
    """Get information about available models."""
    return {
        "available_models": ["svm", "random_forest", "logistic_regression", "ensemble"],
        "default_model": "random_forest",
        "models_trained": detector.is_trained,
        "trained_feature_names": getattr(detector, 'feature_names', None),
        "description": {
            "svm": "Support Vector Machine with RBF kernel",
            "random_forest": "Random Forest with 100 estimators",
            "logistic_regression": "Logistic Regression classifier",
            "ensemble": "Ensemble voting from all three models"
        }
    }


@app.get("/education/tips")
async def get_education_tips():
    """Get educational tips about phishing detection."""
    return {
        "tips": [
            {
                "title": "Check the URL",
                "description": "Phishing URLs often contain misspellings or use suspicious domains. Always verify the domain matches the legitimate website."
            },
            {
                "title": "Look for HTTPS",
                "description": "Legitimate websites use HTTPS (padlock icon). However, some phishing sites also use HTTPS, so this alone isn't enough."
            },
            {
                "title": "Beware of Urgency",
                "description": "Phishing emails often create a sense of urgency to make you act without thinking. Take time to verify."
            },
            {
                "title": "Check the Sender",
                "description": "Verify the sender's email address. Phishers often use addresses that look similar to legitimate ones."
            },
            {
                "title": "Don't Click Suspicious Links",
                "description": "Hover over links to see where they really go before clicking. Be especially cautious with shortened URLs."
            },
            {
                "title": "Watch for Grammar Errors",
                "description": "Phishing messages often contain spelling and grammar mistakes that legitimate companies would not make."
            },
            {
                "title": "Verify Requests Independently",
                "description": "If you receive a request for sensitive information, contact the company directly using a known phone number or website."
            }
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
