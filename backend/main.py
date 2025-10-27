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
        
        # Convert to numpy array (ensure consistent order)
        feature_vector = np.array([
            features.get('url_length', 0),
            features.get('num_dots', 0),
            features.get('num_hyphens', 0),
            features.get('num_underscores', 0),
            features.get('num_slashes', 0),
            features.get('num_questions', 0),
            features.get('num_equals', 0),
            features.get('num_at', 0),
            features.get('num_ampersands', 0),
            features.get('num_digits', 0),
            features.get('has_ip', 0),
            features.get('is_https', 0),
            features.get('domain_length', 0),
            features.get('path_length', 0),
            features.get('query_length', 0),
            features.get('has_suspicious_keywords', 0),
        ])
        
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
        
        # For email, we'll use a subset of features that are relevant
        # In a real application, you'd train separate models for email vs URL
        # Here we'll use the URL feature space for simplicity
        feature_vector = np.array([
            features.get('content_length', 0) / 10,  # Scale down
            features.get('num_urls', 0),
            features.get('num_suspicious_keywords', 0),
            features.get('has_money_keywords', 0) * 10,
            features.get('num_exclamations', 0),
            features.get('capital_ratio', 0) * 100,
            features.get('mentions_attachments', 0) * 10,
            features.get('sender_length', 0),
            features.get('sender_has_numbers', 0) * 10,
            0, 0, 0, 0, 0, 0, 0  # Padding for URL-specific features
        ])
        
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
    Train the ML models on synthetic data.
    """
    try:
        from data.dataset_generator import generate_synthetic_dataset
        
        # Generate dataset
        X, y = generate_synthetic_dataset(n_samples=request.n_samples)
        
        # Train models
        scores = detector.train(X.values, y)
        
        # Save models
        detector.save_models()
        
        return TrainingResponse(
            success=True,
            message=f"Models trained successfully on {request.n_samples} samples",
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
