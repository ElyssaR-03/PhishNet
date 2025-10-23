# PhishNet API Documentation

## Base URL
```
http://localhost:8000
```

## Interactive Documentation
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Endpoints

### Health Check

#### `GET /health`
Check API health status and model availability.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

### URL Analysis

#### `POST /analyze/url`
Analyze a URL for phishing indicators.

**Request Body:**
```json
{
  "url": "https://example.com",
  "model": "random_forest"
}
```

**Parameters:**
- `url` (string, required): The URL to analyze
- `model` (string, optional): Model to use
  - Options: `svm`, `random_forest`, `logistic_regression`, `ensemble`
  - Default: `random_forest`

**Response:**
```json
{
  "is_phishing": false,
  "confidence": 0.95,
  "risk_level": "Safe",
  "features": {
    "url_length": 22,
    "num_dots": 2,
    "has_ip": 0,
    "is_https": 1,
    ...
  },
  "model_used": "random_forest",
  "individual_predictions": null
}
```

**Risk Levels:**
- `Safe`: Not phishing
- `Low Risk`: Possibly phishing (confidence 50-60%)
- `Medium Risk`: Likely phishing (confidence 60-80%)
- `High Risk`: Very likely phishing (confidence 80%+)

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.google.com","model":"ensemble"}'
```

### Email Analysis

#### `POST /analyze/email`
Analyze email content and sender for phishing indicators.

**Request Body:**
```json
{
  "content": "Email body text here...",
  "sender": "sender@example.com",
  "model": "random_forest"
}
```

**Parameters:**
- `content` (string, required): Email body text
- `sender` (string, optional): Sender's email address
- `model` (string, optional): Model to use (same options as URL analysis)

**Response:**
```json
{
  "is_phishing": true,
  "confidence": 0.87,
  "risk_level": "High Risk",
  "features": {
    "content_length": 150,
    "num_urls": 2,
    "num_suspicious_keywords": 5,
    ...
  },
  "model_used": "random_forest",
  "individual_predictions": null
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/analyze/email" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "URGENT! Verify your account NOW!",
    "sender": "noreply@suspicious.com",
    "model": "ensemble"
  }'
```

### Model Training

#### `POST /train`
Train ML models on synthetic dataset.

**Request Body:**
```json
{
  "n_samples": 1000
}
```

**Parameters:**
- `n_samples` (integer, optional): Number of samples to generate
  - Default: 1000
  - Range: 100-10000

**Response:**
```json
{
  "success": true,
  "message": "Models trained successfully on 1000 samples",
  "accuracies": {
    "svm": 0.995,
    "random_forest": 0.995,
    "logistic_regression": 1.0
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/train" \
  -H "Content-Type: application/json" \
  -d '{"n_samples": 1000}'
```

### Model Information

#### `GET /models/info`
Get information about available ML models.

**Response:**
```json
{
  "available_models": [
    "svm",
    "random_forest",
    "logistic_regression",
    "ensemble"
  ],
  "default_model": "random_forest",
  "models_trained": true,
  "description": {
    "svm": "Support Vector Machine with RBF kernel",
    "random_forest": "Random Forest with 100 estimators",
    "logistic_regression": "Logistic Regression classifier",
    "ensemble": "Ensemble voting from all three models"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/models/info
```

### Education Tips

#### `GET /education/tips`
Get phishing education tips and best practices.

**Response:**
```json
{
  "tips": [
    {
      "title": "Check the URL",
      "description": "Phishing URLs often contain misspellings..."
    },
    ...
  ]
}
```

**Example:**
```bash
curl http://localhost:8000/education/tips
```

## Error Responses

### 400 Bad Request
Invalid request parameters.

```json
{
  "detail": "Invalid URL format"
}
```

### 503 Service Unavailable
Models not trained or not available.

```json
{
  "detail": "Models not trained. Please train models first using /train endpoint."
}
```

### 500 Internal Server Error
Server error during processing.

```json
{
  "detail": "Analysis failed: [error message]"
}
```

## Rate Limiting

Currently no rate limiting is implemented. For production use, consider implementing rate limiting based on your requirements.

## CORS

CORS is enabled for all origins in development. For production, configure specific origins in `main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourfrontend.com"],
    ...
)
```

## Authentication

Currently no authentication is implemented. For production use, consider adding:
- API key authentication
- JWT tokens
- OAuth2

## Feature Descriptions

### URL Features
- `url_length`: Total length of URL
- `num_dots`: Number of dots (.)
- `num_hyphens`: Number of hyphens (-)
- `num_underscores`: Number of underscores (_)
- `num_slashes`: Number of slashes (/)
- `num_questions`: Number of question marks (?)
- `num_equals`: Number of equal signs (=)
- `num_at`: Number of @ symbols
- `num_ampersands`: Number of ampersands (&)
- `num_digits`: Number of digits
- `has_ip`: 1 if IP address used, 0 otherwise
- `is_https`: 1 if HTTPS, 0 otherwise
- `domain_length`: Length of domain name
- `path_length`: Length of URL path
- `query_length`: Length of query string
- `has_suspicious_keywords`: 1 if suspicious words found, 0 otherwise

### Email Features
- `content_length`: Length of email content
- `num_urls`: Number of URLs in email
- `num_suspicious_keywords`: Count of suspicious words
- `has_money_keywords`: 1 if money-related terms found
- `num_exclamations`: Number of exclamation marks
- `capital_ratio`: Ratio of capital letters to total
- `mentions_attachments`: 1 if attachments mentioned
- `sender_length`: Length of sender email
- `sender_has_numbers`: 1 if sender has numbers

## Python Client Example

```python
import requests

# Analyze URL
response = requests.post(
    "http://localhost:8000/analyze/url",
    json={"url": "https://example.com", "model": "ensemble"}
)
result = response.json()
print(f"Is phishing: {result['is_phishing']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Risk level: {result['risk_level']}")
```

## JavaScript/Node.js Client Example

```javascript
// Using fetch
const analyzeURL = async (url, model = 'random_forest') => {
  const response = await fetch('http://localhost:8000/analyze/url', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ url, model }),
  });
  return await response.json();
};

// Usage
const result = await analyzeURL('https://example.com', 'ensemble');
console.log('Is phishing:', result.is_phishing);
console.log('Confidence:', result.confidence);
```
