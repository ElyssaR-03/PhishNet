import React, { useState } from 'react';
import { analyzeEmail } from '../services/api';
import './Analyzer.css';

function EmailAnalyzer({ onAnalysisComplete }) {
  const [content, setContent] = useState('');
  const [sender, setSender] = useState('');
  const [model, setModel] = useState('random_forest');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const results = await analyzeEmail(content, sender, model);
      onAnalysisComplete(results);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const loadExample = (type) => {
    if (type === 'legitimate') {
      setContent('Hello, this is a friendly reminder about your upcoming meeting tomorrow at 10 AM. Please let me know if you need to reschedule.');
      setSender('colleague@company.com');
    } else {
      setContent('URGENT! Your account has been suspended! Click here immediately to verify your account and claim your $1000 prize! Act now!!!');
      setSender('noreply123@suspicious-bank.com');
    }
  };

  return (
    <div className="analyzer">
      <h2>📧 Email Analysis</h2>
      <p className="description">
        Analyze email content and sender information for phishing indicators.
      </p>

      <form onSubmit={handleSubmit} className="analyzer-form">
        <div className="form-group">
          <label htmlFor="sender">Sender Email (optional):</label>
          <input
            type="email"
            id="sender"
            value={sender}
            onChange={(e) => setSender(e.target.value)}
            placeholder="sender@example.com"
            className="form-input"
          />
        </div>

        <div className="form-group">
          <label htmlFor="content">Email Content:</label>
          <textarea
            id="content"
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Paste email content here..."
            required
            rows={8}
            className="form-textarea"
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
          {loading ? 'Analyzing...' : 'Analyze Email'}
        </button>
      </form>

      <div className="examples">
        <p className="examples-title">Load example:</p>
        <button
          onClick={() => loadExample('legitimate')}
          className="example-button"
        >
          Legitimate Email
        </button>
        <button
          onClick={() => loadExample('phishing')}
          className="example-button"
        >
          Suspicious Email
        </button>
      </div>
    </div>
  );
}

export default EmailAnalyzer;
