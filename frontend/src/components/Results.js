import React from 'react';
import './Results.css';

function Results({ results }) {
  const { is_phishing, confidence, risk_level, features, model_used, individual_predictions } = results;

  const getRiskColor = (level) => {
    switch (level) {
      case 'High Risk':
        return '#e74c3c';
      case 'Medium Risk':
        return '#f39c12';
      case 'Low Risk':
        return '#f1c40f';
      case 'Safe':
        return '#2ecc71';
      default:
        return '#95a5a6';
    }
  };

  const getRiskIcon = (level) => {
    switch (level) {
      case 'High Risk':
        return '🚨';
      case 'Medium Risk':
        return '⚠️';
      case 'Low Risk':
        return '⚡';
      case 'Safe':
        return '✅';
      default:
        return '❓';
    }
  };

  return (
    <div className="results">
      <h2>Analysis Results</h2>

      <div className="result-card" style={{ borderLeftColor: getRiskColor(risk_level) }}>
        <div className="result-header">
          <span className="result-icon">{getRiskIcon(risk_level)}</span>
          <div className="result-main">
            <h3>{is_phishing ? 'Potential Phishing Detected' : 'Appears Safe'}</h3>
            <p className="risk-level" style={{ color: getRiskColor(risk_level) }}>
              {risk_level}
            </p>
          </div>
        </div>

        <div className="confidence-section">
          <div className="confidence-label">
            <span>Confidence:</span>
            <span className="confidence-value">{(confidence * 100).toFixed(1)}%</span>
          </div>
          <div className="confidence-bar">
            <div
              className="confidence-fill"
              style={{
                width: `${confidence * 100}%`,
                background: getRiskColor(risk_level),
              }}
            />
          </div>
        </div>

        <div className="model-info">
          <p>
            <strong>Model Used:</strong> {model_used.replace('_', ' ').toUpperCase()}
          </p>
        </div>

        {individual_predictions && (
          <div className="ensemble-results">
            <h4>Individual Model Predictions:</h4>
            <div className="predictions-grid">
              {Object.entries(individual_predictions).map(([model, prediction]) => (
                <div key={model} className="prediction-item">
                  <span className="model-name">
                    {model.replace('_', ' ').replace(/\b\w/g, (l) => l.toUpperCase())}
                  </span>
                  <span className={`prediction-badge ${prediction === 1 ? 'phishing' : 'safe'}`}>
                    {prediction === 1 ? 'Phishing' : 'Safe'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="features-section">
          <h4>Extracted Features:</h4>
          <div className="features-grid">
            {Object.entries(features).slice(0, 8).map(([key, value]) => (
              <div key={key} className="feature-item">
                <span className="feature-name">
                  {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}:
                </span>
                <span className="feature-value">
                  {typeof value === 'number' ? value.toFixed(2) : value.toString()}
                </span>
              </div>
            ))}
          </div>
          {Object.keys(features).length > 8 && (
            <details className="more-features">
              <summary>Show all features ({Object.keys(features).length} total)</summary>
              <div className="features-grid">
                {Object.entries(features).slice(8).map(([key, value]) => (
                  <div key={key} className="feature-item">
                    <span className="feature-name">
                      {key.replace(/_/g, ' ').replace(/\b\w/g, (l) => l.toUpperCase())}:
                    </span>
                    <span className="feature-value">
                      {typeof value === 'number' ? value.toFixed(2) : value.toString()}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>

        <div className="warning-box">
          <p>
            <strong>⚠️ Note:</strong> This is an automated analysis tool. Always exercise caution
            and verify suspicious content through official channels.
          </p>
        </div>
      </div>
    </div>
  );
}

export default Results;
