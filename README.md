# PhishNet 🛡️

AI-Powered Phishing Detection System using Machine Learning

## Overview

PhishNet is a comprehensive web application that detects phishing attempts using machine learning classification. It analyzes URLs and email content to identify potential phishing attacks, helping users stay safe online.

## Features

### 🤖 Machine Learning Models
- **Support Vector Machine (SVM)** with RBF kernel
- **Random Forest** with 100 estimators
- **Logistic Regression** classifier
- **Ensemble Mode** combining all three models with majority voting

### 🔍 Analysis Capabilities
- **URL Analysis**: Detects suspicious URLs using 16+ features
- **Email Analysis**: Analyzes email content and sender information
- **Real-time Detection**: Instant results with confidence scores
- **Risk Level Assessment**: Categorizes threats (Safe, Low, Medium, High Risk)
- **PhishTank Integration**: Optional real-time checking against PhishTank's verified phishing database

### 📚 Educational Dashboard
- Phishing statistics and trends
- Protection tips and best practices
- Warning signs identification
- Action steps for suspected phishing

### 🎨 Modern UI
- Clean, responsive React interface
- Interactive analysis forms
- Visual results display with confidence meters
- Mobile-friendly design

## Architecture

### Backend (FastAPI)
- RESTful API with automatic documentation
- Feature extraction engine
- ML model management and training
- CORS-enabled for frontend integration

### Frontend (React)
- Component-based architecture
- Axios for API communication
- Real-time analysis feedback
- Educational content integration

### ML Pipeline
- Feature extraction from URLs and emails
- StandardScaler for feature normalization
- Three separate classifiers
- Model persistence and loading

## Project Structure

```
PhishNet/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── feature_extractor.py    # Feature extraction module
│   ├── train_models.py         # Model training script
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   ├── ml_models.py        # ML models implementation
│   │   └── saved_models/       # Trained model storage
│   ├── data/
│   │   └── dataset_generator.py # Dataset generation
│   └── tests/
│       ├── test_feature_extractor.py
│       ├── test_ml_models.py
│       └── test_api.py
├── frontend/
│   ├── public/
│   │   └── index.html
│   ├── src/
│   │   ├── App.js              # Main application
│   │   ├── components/         # React components
│   │   │   ├── URLAnalyzer.js
│   │   │   ├── EmailAnalyzer.js
│   │   │   ├── EducationDashboard.js
│   │   │   └── Results.js
│   │   └── services/
│   │       └── api.js          # API client
│   └── package.json
└── docs/
    └── uml/
        ├── class_diagram.md    # System class diagram
        └── sequence_diagram.md # Sequence diagrams
```

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- npm or yarn

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure PhishTank API (Optional but Recommended):
```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your PhishTank API key
# Get your free API key from: https://www.phishtank.com/api_info.php
# PHISHTANK_API_KEY=your_actual_api_key_here
```

5. Train the models:
```bash
python train_models.py
```

6. Start the FastAPI server:
```bash
python main.py
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The application will open at `http://localhost:3000`

## Usage

### URL Analysis
1. Navigate to the "URL Analysis" tab
2. Enter a URL to analyze
3. Select an ML model (or ensemble)
4. Click "Analyze URL"
5. View results with confidence score and risk level

### Email Analysis
1. Navigate to the "Email Analysis" tab
2. Enter the sender's email (optional)
3. Paste the email content
4. Select an ML model
5. Click "Analyze Email"
6. Review the analysis results

### Education
1. Navigate to the "Education" tab
2. Browse phishing statistics
3. Read protection tips
4. Learn about warning signs
5. Review best practices

## Configuration

### PhishTank API Integration

PhishNet supports integration with the PhishTank database for real-time verification of URLs against a crowdsourced database of verified phishing sites.

#### Getting a PhishTank API Key

1. Visit [PhishTank API Registration](https://www.phishtank.com/api_info.php)
2. Sign up for a free account
3. Request an API key (free for non-commercial use)
4. Copy your API key

#### Configuring PhishTank

1. In the `backend` directory, copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your API key:
   ```bash
   PHISHTANK_API_KEY=your_actual_api_key_here
   ```

3. Restart the backend server to apply the changes

#### How PhishTank Integration Works

When PhishTank is enabled:
- URLs are checked against the PhishTank verified database
- If a URL is found in PhishTank's database as verified phishing, the result overrides ML prediction
- PhishTank results are included in the API response under `phishtank_check`
- The system works without PhishTank if no API key is configured (ML-only mode)

## API Endpoints

### Analysis
- `POST /analyze/url` - Analyze a URL for phishing
- `POST /analyze/email` - Analyze email content

### Models
- `GET /models/info` - Get information about available models
- `POST /train` - Train models on synthetic data

### Education
- `GET /education/tips` - Get phishing prevention tips

### Health
- `GET /health` - Check API health status

## Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
```

Test coverage includes:
- Feature extraction unit tests
- ML model training and prediction tests
- API endpoint integration tests

### Running Individual Test Suites
```bash
# Test feature extraction
pytest tests/test_feature_extractor.py -v

# Test ML models
pytest tests/test_ml_models.py -v

# Test API endpoints
pytest tests/test_api.py -v
```

## ML Models

### Support Vector Machine (SVM)
- Kernel: RBF (Radial Basis Function)
- Probability estimates enabled
- Good for high-dimensional data

### Random Forest
- 100 decision trees
- Robust to overfitting
- Feature importance analysis

### Logistic Regression
- Maximum iterations: 1000
- Fast training and prediction
- Interpretable coefficients

### Ensemble Method
- Combines all three models
- Majority voting for final prediction
- Average confidence from all models

## Feature Extraction

### URL Features (16 features)
- Length and structure metrics
- Special character counts
- Domain analysis
- HTTPS detection
- IP address detection
- Suspicious keyword matching

### Email Features (9 features)
- Content length and structure
- URL count in email
- Suspicious keyword frequency
- Money-related terms
- Urgency indicators
- Sender analysis

## Security Considerations

- Models should be retrained regularly with real phishing data
- Always verify suspicious content through official channels
- This tool is supplementary - use human judgment
- Report phishing to appropriate authorities
- Keep all dependencies updated

## Performance

- API response time: < 100ms for single prediction
- Model accuracy: ~90%+ on synthetic data
- Supports concurrent requests
- Lightweight model files (~5MB total)

## Future Enhancements

- [ ] Real phishing dataset integration
- [ ] Email header analysis
- [ ] Image-based phishing detection
- [ ] Browser extension
- [ ] Threat intelligence integration
- [ ] User feedback loop for model improvement
- [ ] Multi-language support
- [ ] Advanced analytics dashboard

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with FastAPI and React
- ML models using scikit-learn
- Inspired by real-world phishing detection needs
- Educational content based on cybersecurity best practices

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Review the documentation in `/docs`
- Check API documentation at `/docs` endpoint

## Disclaimer

This tool is for educational and defensive purposes only. While it uses machine learning to detect potential phishing, no automated system is 100% accurate. Always exercise caution and verify suspicious communications through official channels.

---

Made with ❤️ for a safer internet