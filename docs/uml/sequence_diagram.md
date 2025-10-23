# PhishNet Sequence Diagrams

## 1. URL Analysis Flow

```
User → Frontend → API → FeatureExtractor → PhishingDetector → Frontend → User

1. User enters URL in frontend
   │
   ├──→ Frontend: URLAnalyzer component
   │    │
   │    ├──→ API: POST /analyze/url
   │    │    │
   │    │    ├──→ FeatureExtractor: extract_url_features(url)
   │    │    │    └──→ Returns: {url_length, num_dots, has_ip, ...}
   │    │    │
   │    │    ├──→ PhishingDetector: predict(features, model_name)
   │    │    │    │
   │    │    │    ├──→ StandardScaler: transform(features)
   │    │    │    ├──→ Model (SVM/RF/LR): predict(scaled_features)
   │    │    │    ├──→ Model: predict_proba(scaled_features)
   │    │    │    └──→ Returns: (prediction, confidence)
   │    │    │
   │    │    └──→ Returns: AnalysisResponse
   │    │
   │    └──→ Results component displays results
   │
   └──→ User views analysis results
```

## 2. Email Analysis Flow

```
User → Frontend → API → FeatureExtractor → PhishingDetector → Frontend → User

1. User enters email content and sender
   │
   ├──→ Frontend: EmailAnalyzer component
   │    │
   │    ├──→ API: POST /analyze/email
   │    │    │
   │    │    ├──→ FeatureExtractor: extract_email_features(content, sender)
   │    │    │    └──→ Returns: {content_length, num_urls, num_suspicious_keywords, ...}
   │    │    │
   │    │    ├──→ PhishingDetector: predict(features, model_name)
   │    │    │    └──→ Returns: (prediction, confidence)
   │    │    │
   │    │    └──→ Returns: AnalysisResponse
   │    │
   │    └──→ Results component displays results
   │
   └──→ User views analysis results
```

## 3. Ensemble Prediction Flow

```
User → Frontend → API → PhishingDetector (multiple models) → Frontend → User

1. User selects "ensemble" model
   │
   ├──→ API: POST /analyze/url (model="ensemble")
   │    │
   │    ├──→ PhishingDetector: predict_ensemble(features)
   │    │    │
   │    │    ├──→ SVM Model: predict(features)
   │    │    │    └──→ prediction_svm
   │    │    │
   │    │    ├──→ Random Forest: predict(features)
   │    │    │    └──→ prediction_rf
   │    │    │
   │    │    ├──→ Logistic Regression: predict(features)
   │    │    │    └──→ prediction_lr
   │    │    │
   │    │    ├──→ Majority voting (if 2+ agree)
   │    │    │
   │    │    └──→ Returns: (ensemble_pred, avg_confidence, individual_preds)
   │    │
   │    └──→ Returns: AnalysisResponse with individual_predictions
   │
   └──→ User views ensemble results with all model predictions
```

## 4. Model Training Flow

```
Admin → Frontend → API → DatasetGenerator → PhishingDetector → API → Frontend

1. Admin requests model training
   │
   ├──→ API: POST /train
   │    │
   │    ├──→ DatasetGenerator: generate_synthetic_dataset(n_samples)
   │    │    │
   │    │    ├──→ Generates legitimate URL patterns
   │    │    ├──→ Generates phishing URL patterns
   │    │    ├──→ Combines and shuffles data
   │    │    │
   │    │    └──→ Returns: (X, y)
   │    │
   │    ├──→ PhishingDetector: train(X, y)
   │    │    │
   │    │    ├──→ Split data (train/test)
   │    │    ├──→ StandardScaler: fit_transform(X_train)
   │    │    │
   │    │    ├──→ SVM: fit(X_train, y_train)
   │    │    ├──→ Random Forest: fit(X_train, y_train)
   │    │    ├──→ Logistic Regression: fit(X_train, y_train)
   │    │    │
   │    │    ├──→ Evaluate on test set
   │    │    │
   │    │    └──→ Returns: accuracy scores
   │    │
   │    ├──→ PhishingDetector: save_models()
   │    │    └──→ Saves models to disk
   │    │
   │    └──→ Returns: TrainingResponse with accuracies
   │
   └──→ Admin views training results
```

## 5. Education Dashboard Flow

```
User → Frontend → API → Frontend → User

1. User navigates to Education tab
   │
   ├──→ Frontend: EducationDashboard component
   │    │
   │    ├──→ API: GET /education/tips
   │    │    └──→ Returns: education tips array
   │    │
   │    └──→ Displays tips, statistics, warning signs, best practices
   │
   └──→ User learns about phishing prevention
```

## 6. System Initialization Flow

```
System Startup → Backend → Models

1. FastAPI application starts
   │
   ├──→ Initialize PhishingDetector
   │    │
   │    ├──→ Create model instances (SVM, RF, LR)
   │    ├──→ Initialize StandardScaler
   │    │
   │    └──→ Try to load pre-trained models
   │         ├──→ If models exist: load from disk
   │         └──→ If not: is_trained = False
   │
   ├──→ Initialize FeatureExtractor
   │
   └──→ API endpoints ready to accept requests
```
