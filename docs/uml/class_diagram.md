# PhishNet Class Diagram

```
┌─────────────────────────────────────┐
│      FeatureExtractor               │
├─────────────────────────────────────┤
│ + extract_url_features(url: str)    │
│   : Dict[str, float]                │
│ + extract_email_features(           │
│     content: str, sender: str)      │
│   : Dict[str, float]                │
│ + get_feature_names()               │
│   : List[str]                       │
└─────────────────────────────────────┘
            │
            │ uses
            ▼
┌─────────────────────────────────────┐
│      PhishingDetector               │
├─────────────────────────────────────┤
│ - models: Dict                      │
│ - scaler: StandardScaler            │
│ - is_trained: bool                  │
│ - model_dir: str                    │
├─────────────────────────────────────┤
│ + train(X, y, test_size)            │
│   : Dict[str, float]                │
│ + predict(X, model_name)            │
│   : Tuple[int, float]               │
│ + predict_ensemble(X)               │
│   : Tuple[int, float, Dict]         │
│ + save_models(prefix)               │
│ + load_models(prefix): bool         │
└─────────────────────────────────────┘
            │
            │ contains
            ▼
┌─────────────────────────────────────┐
│      ML Models                      │
├─────────────────────────────────────┤
│ - SVC (Support Vector Machine)      │
│ - RandomForestClassifier            │
│ - LogisticRegression                │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      DatasetGenerator               │
├─────────────────────────────────────┤
│ + generate_synthetic_dataset(       │
│     n_samples: int)                 │
│   : Tuple[DataFrame, ndarray]       │
│ + save_dataset(X, y, filepath)      │
│ + load_dataset(filepath)            │
│   : Tuple[DataFrame, ndarray]       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│      FastAPI Application            │
├─────────────────────────────────────┤
│ - detector: PhishingDetector        │
│ - feature_extractor:                │
│   FeatureExtractor                  │
├─────────────────────────────────────┤
│ + analyze_url(request)              │
│   : AnalysisResponse                │
│ + analyze_email(request)            │
│   : AnalysisResponse                │
│ + train_models(request)             │
│   : TrainingResponse                │
│ + get_models_info()                 │
│ + get_education_tips()              │
│ + health_check()                    │
└─────────────────────────────────────┘

Request/Response Models:
┌─────────────────────────────────────┐
│   URLAnalysisRequest                │
├─────────────────────────────────────┤
│ + url: str                          │
│ + model: str                        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│   EmailAnalysisRequest              │
├─────────────────────────────────────┤
│ + content: str                      │
│ + sender: Optional[str]             │
│ + model: str                        │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│   AnalysisResponse                  │
├─────────────────────────────────────┤
│ + is_phishing: bool                 │
│ + confidence: float                 │
│ + risk_level: str                   │
│ + features: Dict[str, float]        │
│ + model_used: str                   │
│ + individual_predictions:           │
│   Optional[Dict[str, int]]          │
└─────────────────────────────────────┘
```

## Component Relationships

1. **FeatureExtractor**: Standalone utility class for extracting features from URLs and emails
2. **PhishingDetector**: Main ML component that manages three models (SVM, RF, LR)
3. **DatasetGenerator**: Utility for creating training datasets
4. **FastAPI Application**: Web service layer that coordinates all components
5. **Pydantic Models**: Request/response schemas for API validation
