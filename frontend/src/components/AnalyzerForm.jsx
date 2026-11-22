import React from 'react';
import { Send, Sliders, FileText, Loader2, RefreshCw } from 'lucide-react';
import ModelSelector from './ModelSelector';

export default function AnalyzerForm({
  text,
  setText,
  selectedModel,
  setSelectedModel,
  models,
  examples,
  confidenceThreshold,
  setConfidenceThreshold,
  onSubmit,
  isLoading,
  error,
}) {
  const handleLoadExample = (exampleText) => {
    setText(exampleText);
  };

  const handleClear = () => {
    setText('');
  };

  const charCount = text.length;

  return (
    <form className="card analyzer-form" onSubmit={onSubmit} aria-labelledby="form-heading">
      <div className="form-header">
        <div>
          <h2 id="form-heading" className="section-title">
            Analyze Online Text
          </h2>
          <p className="section-subtitle">
            Paste any anonymized post or select a realistic sample to evaluate distress linguistic cues and uncertainty.
          </p>
        </div>

        {text && (
          <button
            type="button"
            className="btn-clear"
            onClick={handleClear}
            disabled={isLoading}
            aria-label="Clear input text"
          >
            <RefreshCw size={14} aria-hidden="true" />
            Clear
          </button>
        )}
      </div>

      {/* Example Presets */}
      <div className="examples-section">
        <span className="examples-label">
          <FileText size={14} aria-hidden="true" /> Sample Scenarios:
        </span>
        <div className="examples-buttons">
          {examples.map((ex) => (
            <button
              key={ex.id}
              type="button"
              className="example-pill-btn"
              onClick={() => handleLoadExample(ex.text)}
              disabled={isLoading}
              title={ex.category}
            >
              {ex.title}
            </button>
          ))}
        </div>
      </div>

      {/* Text Area Input */}
      <div className="form-group">
        <div className="label-row">
          <label htmlFor="post-content" className="form-label">
            Post Content <span className="required-marker" aria-hidden="true">*</span>
          </label>
          <span className="char-count" aria-live="polite">
            {charCount} characters
          </span>
        </div>

        <textarea
          id="post-content"
          name="post_content"
          className="post-textarea"
          rows={5}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter anonymized social media text or post... (e.g., 'I feel completely overwhelmed, exhausted, and hopelessly trapped.')"
          required
          disabled={isLoading}
          aria-describedby="textarea-hint"
        />
        <p id="textarea-hint" className="field-hint">
          Preprocessing will automatically strip URLs, Reddit handles (r/..., u/...), and HTML artifacts for privacy preservation.
        </p>
      </div>

      {/* Model Selector */}
      <ModelSelector
        models={models}
        selectedModel={selectedModel}
        onSelectModel={setSelectedModel}
        disabled={isLoading}
      />

      {/* Abstention Threshold Slider */}
      <div className="form-group threshold-group">
        <div className="label-row">
          <label htmlFor="confidence-slider" className="form-label slider-label">
            <Sliders size={16} aria-hidden="true" />
            <span>Selective Abstention Confidence Threshold: <strong>{(confidenceThreshold * 100).toFixed(0)}%</strong></span>
          </label>
          <span className="threshold-indicator">
            {confidenceThreshold >= 0.80 ? 'Strict (High Abstention)' : confidenceThreshold >= 0.65 ? 'Balanced' : 'Permissive'}
          </span>
        </div>

        <input
          type="range"
          id="confidence-slider"
          min="0.50"
          max="0.95"
          step="0.05"
          value={confidenceThreshold}
          onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
          disabled={isLoading}
          className="threshold-slider"
          aria-valuemin="0.50"
          aria-valuemax="0.95"
          aria-valuenow={confidenceThreshold}
          aria-valuetext={`${(confidenceThreshold * 100).toFixed(0)}% confidence threshold`}
        />

        <div className="slider-ticks" aria-hidden="true">
          <span>50% (No Abstain)</span>
          <span>70% (Default)</span>
          <span>85% (Conservative)</span>
          <span>95% (Extreme)</span>
        </div>

        <p className="field-hint">
          If model certainty max(P, 1-P) falls below this threshold, the model abstains rather than making an unconfident prediction.
        </p>
      </div>

      {/* Error Alert if any */}
      {error && (
        <div className="error-alert" role="alert">
          {error}
        </div>
      )}

      {/* Submit Button */}
      <div className="submit-row">
        <button
          type="submit"
          className="btn-primary"
          disabled={isLoading || !text.trim()}
        >
          {isLoading ? (
            <>
              <Loader2 className="spinner" size={18} aria-hidden="true" />
              <span>Analyzing Post & Uncertainty...</span>
            </>
          ) : (
            <>
              <Send size={18} aria-hidden="true" />
              <span>Classify & Quantify Uncertainty</span>
            </>
          )}
        </button>
      </div>
    </form>
  );
}
