import React, { useState } from 'react';
import { analyzeURL } from '../services/api';
import './Analyzer.css';

function URLAnalyzer({ onAnalysisComplete }) {
  const [url, setUrl] = useState('');
  const [model, setModel] = useState('random_forest');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const results = await analyzeURL(url, model);
      onAnalysisComplete(results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="analyzer">
      <h2>🔗 URL Analysis</h2>
      <p className="description">
        Enter a URL to check if it's potentially a phishing website.
      </p>

      <form onSubmit={handleSubmit} className="analyzer-form">
        <div className="form-group">
          <label htmlFor="url">URL to Analyze:</label>
          <input
            type="text"
            id="url"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://example.com"
            required
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label htmlFor="model">ML Model:</label>
          <select
            id="model"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            className="form-select"
          >
            <option value="random_forest">Random Forest</option>
            <option value="svm">Support Vector Machine</option>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="ensemble">Ensemble (All Models)</option>
          </select>
        </div>

        {error && <div className="error-message">{error}</div>}

        <button type="submit" disabled={loading} className="submit-button">
          {loading ? 'Analyzing...' : 'Analyze URL'}
        </button>
      </form>

      <div className="examples">
        <p className="examples-title">Try these examples:</p>
        <button
          onClick={() => setUrl('https://www.google.com')}
          className="example-button"
        >
          Legitimate: google.com
        </button>
        <button
          onClick={() => setUrl('http://192.168.1.1/login-verify-account-security')}
          className="example-button"
        >
          Suspicious: IP with keywords
        </button>
      </div>
    </div>
  );
}

export default URLAnalyzer;
