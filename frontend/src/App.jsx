import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import AnalyzerForm from './components/AnalyzerForm';
import ResultCard from './components/ResultCard';
import BenchmarkTable from './components/BenchmarkTable';
import AbstentionVisualizer from './components/AbstentionVisualizer';
import { Activity, BarChart2, Sliders } from 'lucide-react';
import './App.css';

export default function App() {
  const [activeTab, setActiveTab] = useState('analyzer'); // 'analyzer' | 'benchmarks' | 'abstention'
  
  // Data states
  const [models, setModels] = useState([]);
  const [examples, setExamples] = useState([]);
  const [metrics, setMetrics] = useState([]);
  const [abstentionData, setAbstentionData] = useState([]);
  
  // Form states
  const [text, setText] = useState('');
  const [selectedModel, setSelectedModel] = useState('TF-IDF + Logistic Regression');
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.70);
  
  // Interaction states
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  // Fetch initial configuration and benchmarks
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        const [modelsRes, examplesRes, metricsRes, abstentionRes] = await Promise.all([
          fetch('/api/models'),
          fetch('/api/examples'),
          fetch('/api/metrics'),
          fetch('/api/abstention-sweep'),
        ]);

        if (modelsRes.ok) {
          const modelsData = await modelsRes.json();
          setModels(modelsData);
        }
        if (examplesRes.ok) {
          const examplesData = await examplesRes.json();
          setExamples(examplesData);
        }
        if (metricsRes.ok) {
          const metricsData = await metricsRes.json();
          setMetrics(metricsData);
        }
        if (abstentionRes.ok) {
          const abstentionJson = await abstentionRes.json();
          setAbstentionData(abstentionJson);
        }
      } catch (err) {
        console.error('Failed to load initial backend data:', err);
      }
    };

    fetchInitialData();
  }, []);

  const handleClassify = async (e) => {
    if (e) e.preventDefault();
    if (!text.trim()) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: text.trim(),
          model_name: selectedModel,
          confidence_threshold: confidenceThreshold,
        }),
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.detail || 'Prediction request failed.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred while communicating with the backend API.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <Header />

      {/* Navigation Tabs */}
      <nav className="tab-nav" aria-label="Main application sections">
        <button
          type="button"
          className={`tab-btn ${activeTab === 'analyzer' ? 'active' : ''}`}
          onClick={() => setActiveTab('analyzer')}
          aria-selected={activeTab === 'analyzer'}
          role="tab"
        >
          <Activity size={18} aria-hidden="true" />
          <span>Interactive Text Analyzer</span>
        </button>

        <button
          type="button"
          className={`tab-btn ${activeTab === 'benchmarks' ? 'active' : ''}`}
          onClick={() => setActiveTab('benchmarks')}
          aria-selected={activeTab === 'benchmarks'}
          role="tab"
        >
          <BarChart2 size={18} aria-hidden="true" />
          <span>Model Benchmark Dashboard</span>
        </button>

        <button
          type="button"
          className={`tab-btn ${activeTab === 'abstention' ? 'active' : ''}`}
          onClick={() => setActiveTab('abstention')}
          aria-selected={activeTab === 'abstention'}
          role="tab"
        >
          <Sliders size={18} aria-hidden="true" />
          <span>Uncertainty & Abstention Curve</span>
        </button>
      </nav>

      {/* Main Tab Content */}
      <main className="tab-content">
        {activeTab === 'analyzer' && (
          <div className="analyzer-layout">
            <AnalyzerForm
              text={text}
              setText={setText}
              selectedModel={selectedModel}
              setSelectedModel={setSelectedModel}
              models={models}
              examples={examples}
              confidenceThreshold={confidenceThreshold}
              setConfidenceThreshold={setConfidenceThreshold}
              onSubmit={handleClassify}
              isLoading={isLoading}
              error={error}
            />

            <ResultCard result={result} />
          </div>
        )}

        {activeTab === 'benchmarks' && (
          <BenchmarkTable metrics={metrics} />
        )}

        {activeTab === 'abstention' && (
          <AbstentionVisualizer data={abstentionData} />
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <div className="footer-content">
          <p>
            <strong>Responsible AI Distress Language Classifier</strong> • Built with FastAPI (Python) & React (Vite)
          </p>
          <p className="footer-sub">
            Pretrained on Dreaddit • TF-IDF + Logistic Regression, Linear SVM, Random Forest & Naive Bayes
          </p>
        </div>
      </footer>
    </div>
  );
}
