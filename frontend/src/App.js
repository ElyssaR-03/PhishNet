import React, { useState } from 'react';
import './App.css';
import URLAnalyzer from './components/URLAnalyzer';
import EmailAnalyzer from './components/EmailAnalyzer';
import EducationDashboard from './components/EducationDashboard';
import Results from './components/Results';

function App() {
  const [activeTab, setActiveTab] = useState('url');
  const [results, setResults] = useState(null);

  const handleAnalysisComplete = (analysisResults) => {
    setResults(analysisResults);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>PhishNet</h1>
        <p className="tagline">AI-Powered Phishing Detection System</p>
      </header>

      <div className="container">
        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'url' ? 'active' : ''}`}
            onClick={() => setActiveTab('url')}
          >
            URL Analysis
          </button>
          <button
            className={`nav-tab ${activeTab === 'email' ? 'active' : ''}`}
            onClick={() => setActiveTab('email')}
          >
            Email Analysis
          </button>
          <button
            className={`nav-tab ${activeTab === 'education' ? 'active' : ''}`}
            onClick={() => setActiveTab('education')}
          >
            Education
          </button>
        </nav>

        <div className="content">
          {activeTab === 'url' && (
            <URLAnalyzer onAnalysisComplete={handleAnalysisComplete} />
          )}
          {activeTab === 'email' && (
            <EmailAnalyzer onAnalysisComplete={handleAnalysisComplete} />
          )}
          {activeTab === 'education' && <EducationDashboard />}
        </div>

        {results && activeTab !== 'education' && (
          <Results results={results} />
        )}
      </div>

      <footer className="App-footer">
        <p>© 2025 PhishNet - Protecting you from phishing attacks</p>
      </footer>
    </div>
  );
}

export default App;
