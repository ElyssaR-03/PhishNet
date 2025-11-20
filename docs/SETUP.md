# PhishNet Setup Guide

Complete guide to setting up and running PhishNet locally.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.8 or higher** - [Download Python](https://www.python.org/downloads/)
- **Node.js 16 or higher** - [Download Node.js](https://nodejs.org/)
- **npm or yarn** - Comes with Node.js
- **Git** - [Download Git](https://git-scm.com/)

### Verify Installation

```bash
python --version  # Should show Python 3.8+
node --version    # Should show v16+
npm --version     # Should show 8+
```

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ElyssaR-03/PhishNet.git
cd PhishNet
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Train the ML models
python train_models.py

# Start the backend server
python main.py
```

The API will be available at `http://localhost:8000`

### 3. Frontend Setup

Open a new terminal (keep backend running):

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The application will open at `http://localhost:3000`

## Detailed Setup Instructions

### Backend Setup (Detailed)

#### Step 1: Create Virtual Environment

A virtual environment isolates your Python dependencies:

```bash
cd backend
python -m venv venv
```

#### Step 2: Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal.

#### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- FastAPI - Web framework
- Uvicorn - ASGI server
- Scikit-learn - Machine learning
- Pandas - Data manipulation
- NumPy - Numerical computing
- Pytest - Testing framework

#### Step 4: Train ML Models

```bash
python train_models.py
```

This will:
1. Generate a synthetic dataset (1000 samples)
2. Train three ML models (SVM, Random Forest, Logistic Regression)
3. Save trained models to `models/saved_models/`
4. Display accuracy metrics

Expected output:
```
PhishNet Model Training
==================================================
Generating synthetic dataset...
Dataset saved to .../phishing_dataset.csv

Dataset size: 1000 samples
Legitimate samples: 500
Phishing samples: 500

Training models...

Model Accuracies:
--------------------------------------------------
Svm: 0.9950 (99.50%)
Random Forest: 0.9950 (99.50%)
Logistic Regression: 1.0000 (100.00%)

Saving trained models...
Models saved successfully!
```

#### Training behavior and custom datasets

- The training script (`backend/train_models.py`) now scans `backend/data/` for CSV files and will include
  any CSV that contains a label column. The loader recognizes common label names such as `is_phishing`,
  `label`, `phishing`, `target`, `y`, `class`, `CLASS_LABEL`, etc., and will normalize string or
  numeric encodings into binary 0/1 labels used by the models. If a CSV contains a recognized label column it will
  be concatenated with other valid CSVs and used for training.

- If your CSV uses a different label name, either rename the column to one of the common names above or
  the loader will often match by substring (e.g., `CLASS_LABEL` will be detected). If normalization fails
  for a particular file the script will skip that file and print a warning.

Example (run training from project root):

```powershell
& "c:\Users\mslys\OneDrive - Tennessee State University\PhishNet\PhishNet\backend\venv\Scripts\python.exe" "c:\Users\mslys\OneDrive - Tennessee State University\PhishNet\PhishNet\backend\train_models.py"
```


#### Step 5: Run Backend Tests

```bash
pytest tests/ -v
```

Expected: All 26 tests should pass.

#### Step 6: Start Backend Server

```bash
python main.py
```

Server will start on `http://localhost:8000`

Access API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Frontend Setup (Detailed)

#### Step 1: Install Node Dependencies

```bash
cd frontend
npm install
```

This installs:
- React - UI framework
- React DOM - React renderer
- Axios - HTTP client
- React Scripts - Development tools

#### Step 2: Configure API Endpoint (Optional)

By default, the frontend connects to `http://localhost:8000`.

To change this, create `.env.local`:

```bash
# frontend/.env.local
REACT_APP_API_URL=http://your-api-url:8000
```

#### Step 3: Start Development Server

```bash
npm start
```

This will:
1. Start webpack dev server
2. Open browser at `http://localhost:3000`
3. Enable hot module reloading

## Verification Steps

### 1. Test Backend API

```bash
# Health check
curl http://localhost:8000/health

# Analyze a URL
curl -X POST "http://localhost:8000/analyze/url" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.google.com","model":"random_forest"}'
```

### 2. Test Frontend

1. Open `http://localhost:3000` in browser
2. Navigate to "URL Analysis" tab
3. Enter a URL (e.g., `https://www.google.com`)
4. Click "Analyze URL"
5. Verify results appear

## Production Deployment

### Backend Deployment

#### Using Docker (Recommended)

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train_models.py

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t phishnet-backend .
docker run -p 8000:8000 phishnet-backend
```

#### Using Systemd (Linux)

Create `/etc/systemd/system/phishnet.service`:

```ini
[Unit]
Description=PhishNet Backend
After=network.target

[Service]
User=www-data
WorkingDirectory=/path/to/PhishNet/backend
Environment="PATH=/path/to/venv/bin"
ExecStart=/path/to/venv/bin/python main.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable phishnet
sudo systemctl start phishnet
```

### Frontend Deployment

#### Build Production Bundle

```bash
cd frontend
npm run build
```

This creates optimized files in `frontend/build/`

#### Deploy to Static Hosting

**Netlify:**
```bash
npm install -g netlify-cli
netlify deploy --prod --dir=build
```

**Vercel:**
```bash
npm install -g vercel
vercel --prod
```

**Traditional Web Server:**
Copy `build/` contents to web server:
```bash
scp -r build/* user@server:/var/www/html/
```

## Troubleshooting

### Backend Issues

#### Port Already in Use

```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8000 | xargs kill -9
```

#### Import Errors

Make sure virtual environment is activated:
```bash
which python  # Should point to venv
pip list      # Should show installed packages
```

#### Model Not Training

Check Python version:
```bash
python --version  # Should be 3.8+
```

Reinstall scikit-learn:
```bash
pip install --upgrade scikit-learn
```

### Frontend Issues

#### npm install Fails

Clear npm cache:
```bash
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### Port 3000 Already in Use

Set custom port:
```bash
# Windows
set PORT=3001 && npm start

# macOS/Linux
PORT=3001 npm start
```

#### API Connection Failed

Verify backend is running:
```bash
curl http://localhost:8000/health
```

Check browser console for CORS errors.

## Environment Variables

### Backend (.env)

```bash
# Optional configuration
MODEL_DIR=models/saved_models
DATASET_DIR=data
LOG_LEVEL=INFO
```

### Frontend (.env.local)

```bash
# API endpoint
REACT_APP_API_URL=http://localhost:8000

# Optional
REACT_APP_ENABLE_ANALYTICS=false
```

## Development Tips

### Backend Development

- Use `uvicorn main:app --reload` for auto-reload
- Check logs in terminal
- Use `/docs` endpoint for API testing
- Run tests frequently: `pytest tests/ -v`

### Frontend Development

- Changes auto-reload in browser
- Check browser console for errors
- Use React DevTools extension
- Test on different screen sizes

## Performance Optimization

### Backend

- Enable caching for predictions
- Use Redis for session storage
- Implement rate limiting
- Use gunicorn with multiple workers:
  ```bash
  gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
  ```

### Frontend

- Enable production build: `npm run build`
- Use CDN for static assets
- Enable gzip compression
- Implement lazy loading

## Security Considerations

### Backend

1. **Authentication**: Add API keys or JWT
2. **Rate Limiting**: Prevent abuse
3. **Input Validation**: Already implemented via Pydantic
4. **CORS**: Configure allowed origins
5. **HTTPS**: Use in production

### Frontend

1. **Environment Variables**: Never commit secrets
2. **Content Security Policy**: Add CSP headers
3. **XSS Protection**: React handles this
4. **HTTPS**: Always use in production

## Next Steps

1. ✅ Backend and frontend running
2. ✅ Tests passing
3. ✅ Models trained
4. 🔄 Customize for your needs
5. 🔄 Add real phishing datasets
6. 🔄 Deploy to production

## Getting Help

- **Documentation**: Check `/docs` directory
- **API Docs**: `http://localhost:8000/docs`
- **Issues**: Open GitHub issue
- **Logs**: Check terminal output

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Scikit-learn Guide](https://scikit-learn.org/stable/)
- [Machine Learning Best Practices](https://ml-ops.org/)
