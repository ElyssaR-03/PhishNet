import React, { useState, useEffect } from 'react';
import { getEducationTips } from '../services/api';
import './EducationDashboard.css';

function EducationDashboard() {
  const [tips, setTips] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTips = async () => {
      try {
        const data = await getEducationTips();
        setTips(data.tips);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchTips();
  }, []);

  if (loading) {
    return <div className="loading">Loading education materials...</div>;
  }

  if (error) {
    return <div className="error-message">Error: {error}</div>;
  }

  return (
    <div className="education-dashboard">
      <h2>📚 Phishing Education</h2>
      <p className="intro">
        Learn how to identify and protect yourself from phishing attacks. Understanding these
        patterns can help you stay safe online.
      </p>

      <div className="stats-section">
        <div className="stat-card">
          <div className="stat-number">3.4B+</div>
          <div className="stat-label">Phishing emails sent daily</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">90%</div>
          <div className="stat-label">Data breaches start with phishing</div>
        </div>
        <div className="stat-card">
          <div className="stat-number">$17.7M</div>
          <div className="stat-label">Average cost of a data breach</div>
        </div>
      </div>

      <div className="tips-section">
        <h3>🛡️ Protection Tips</h3>
        <div className="tips-grid">
          {tips.map((tip, index) => (
            <div key={index} className="tip-card">
              <div className="tip-number">{index + 1}</div>
              <h4>{tip.title}</h4>
              <p>{tip.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="warning-signs">
        <h3>🚩 Common Warning Signs</h3>
        <div className="signs-grid">
          <div className="sign-item">
            <span className="sign-icon">⚠️</span>
            <div>
              <h4>Urgent Language</h4>
              <p>Claims of account suspension or immediate action required</p>
            </div>
          </div>
          <div className="sign-item">
            <span className="sign-icon">🔗</span>
            <div>
              <h4>Suspicious Links</h4>
              <p>URLs that don't match the claimed sender or contain misspellings</p>
            </div>
          </div>
          <div className="sign-item">
            <span className="sign-icon">✉️</span>
            <div>
              <h4>Generic Greetings</h4>
              <p>No personalization or uses generic terms like "Dear Customer"</p>
            </div>
          </div>
          <div className="sign-item">
            <span className="sign-icon">📎</span>
            <div>
              <h4>Unexpected Attachments</h4>
              <p>Unsolicited files, especially .exe, .zip, or .scr files</p>
            </div>
          </div>
          <div className="sign-item">
            <span className="sign-icon">💰</span>
            <div>
              <h4>Too Good to Be True</h4>
              <p>Promises of prizes, money, or unrealistic offers</p>
            </div>
          </div>
          <div className="sign-item">
            <span className="sign-icon">📧</span>
            <div>
              <h4>Sender Mismatch</h4>
              <p>Email address doesn't match the organization it claims to be from</p>
            </div>
          </div>
        </div>
      </div>

      <div className="best-practices">
        <h3>✅ Best Practices</h3>
        <ul className="practices-list">
          <li>Enable two-factor authentication (2FA) on all accounts</li>
          <li>Keep software and operating systems updated</li>
          <li>Use a password manager with unique passwords</li>
          <li>Verify sender identity through independent channels</li>
          <li>Report suspicious emails to your IT department or email provider</li>
          <li>Never enter credentials on a site reached via email link</li>
          <li>Hover over links before clicking to see the actual destination</li>
          <li>Be skeptical of unsolicited communications requesting information</li>
        </ul>
      </div>

      <div className="action-section">
        <h3>🎯 What to Do If You're Targeted</h3>
        <div className="action-steps">
          <div className="step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Don't Click</h4>
              <p>Don't click any links or download attachments</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Report It</h4>
              <p>Forward to your IT team or report to authorities</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Delete</h4>
              <p>Remove the suspicious email from your inbox</p>
            </div>
          </div>
          <div className="step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Change Passwords</h4>
              <p>If you clicked, change passwords immediately</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default EducationDashboard;
