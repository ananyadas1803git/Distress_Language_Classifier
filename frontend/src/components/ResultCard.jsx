import React from 'react';
import {
  AlertTriangle,
  AlertOctagon,
  CheckCircle2,
  HelpCircle,
  Hash,
  HeartHandshake,
  ShieldAlert,
} from 'lucide-react';

export default function ResultCard({ result }) {
  if (!result) return null;

  const {
    text,
    cleaned_text,
    model_name,
    probability,
    confidence,
    confidence_threshold,
    is_abstained,
    decision,
    top_tokens,
    ethical_notice,
  } = result;

  const probPercent = (probability * 100).toFixed(1);
  const confPercent = (confidence * 100).toFixed(1);
  const threshPercent = (confidence_threshold * 100).toFixed(0);

  let statusConfig = {
    badgeClass: 'badge-distress',
    icon: <AlertOctagon size={20} aria-hidden="true" />,
    title: 'Distress-Related Language Signals Detected',
    explanation:
      'The statistical frequency of terms in this post closely aligns with patterns identified in distress-related online corpora.',
  };

  if (is_abstained) {
    statusConfig = {
      badgeClass: 'badge-abstained',
      icon: <AlertTriangle size={20} aria-hidden="true" />,
      title: 'Classifier Abstained (High Uncertainty)',
      explanation: `Model certainty (${confPercent}%) fell below your configured threshold (${threshPercent}%). The system responsibly abstains rather than making an unreliable classification on ambiguous text.`,
    };
  } else if (decision.includes('No Distress')) {
    statusConfig = {
      badgeClass: 'badge-neutral',
      icon: <CheckCircle2 size={20} aria-hidden="true" />,
      title: 'No Distress Language Detected',
      explanation:
        'The linguistic markers and vocabulary in this text do not statistically match distress language thresholds.',
    };
  }

  return (
    <article className="card result-card" aria-labelledby="result-heading">
      <div className="result-header">
        <div>
          <span className="result-model-badge">Evaluated with {model_name}</span>
          <h3 id="result-heading" className="result-title">
            Analysis & Uncertainty Results
          </h3>
        </div>

        <div className={`status-badge-lg ${statusConfig.badgeClass}`}>
          {statusConfig.icon}
          <span>{decision}</span>
        </div>
      </div>

      <p className="status-explanation">{statusConfig.explanation}</p>

      {/* Probability & Confidence Meters */}
      <div className="metrics-grid">
        <div className="metric-box">
          <div className="metric-box-header">
            <span className="metric-box-label">Distress Language Probability</span>
            <span className="metric-box-val">{probPercent}%</span>
          </div>
          <div className="meter-track">
            <div
              className={`meter-fill ${probability >= 0.5 ? 'fill-distress' : 'fill-neutral'}`}
              style={{ width: `${Math.min(Math.max(probability * 100, 3), 100)}%` }}
              role="progressbar"
              aria-valuenow={probability * 100}
              aria-valuemin="0"
              aria-valuemax="100"
            />
          </div>
          <span className="meter-subtext">P(Distress | TF-IDF features)</span>
        </div>

        <div className="metric-box">
          <div className="metric-box-header">
            <span className="metric-box-label">Decision Certainty vs Threshold</span>
            <span className="metric-box-val">{confPercent}%</span>
          </div>
          <div className="meter-track">
            <div
              className={`meter-fill ${is_abstained ? 'fill-abstained' : 'fill-blue'}`}
              style={{ width: `${Math.min(Math.max(confidence * 100, 3), 100)}%` }}
              role="progressbar"
              aria-valuenow={confidence * 100}
              aria-valuemin="0"
              aria-valuemax="100"
            />
          </div>
          <span className="meter-subtext">
            Threshold: {threshPercent}% | Status: {is_abstained ? 'Uncertain (Abstained)' : 'Confident (Decided)'}
          </span>
        </div>
      </div>

      {/* Linguistic Token Attributions */}
      {top_tokens && top_tokens.length > 0 && (
        <section className="tokens-section" aria-labelledby="tokens-heading">
          <h4 id="tokens-heading" className="subsection-title">
            <Hash size={16} aria-hidden="true" />
            <span>Key Contributing Linguistic Tokens (Feature Attribution)</span>
          </h4>
          <p className="subsection-desc">
            Tokens identified by the model that exerted the strongest statistical influence on the score:
          </p>
          <div className="tokens-list">
            {top_tokens.map((t, idx) => (
              <span
                key={idx}
                className={`token-chip ${t.direction === 'distress' ? 'token-distress' : 'token-neutral'}`}
                title={`Weight: ${t.weight > 0 ? '+' : ''}${t.weight}`}
              >
                <strong>{t.token}</strong>
                <span className="token-weight">
                  {t.weight > 0 ? `+${t.weight.toFixed(3)}` : t.weight.toFixed(3)}
                </span>
              </span>
            ))}
          </div>
        </section>
      )}

      {/* Anonymized Text Preview */}
      <details className="anonymized-details">
        <summary className="anonymized-summary">
          <span>View Anonymized & Preprocessed Text (Handles/URLs Stripped)</span>
        </summary>
        <div className="anonymized-body">
          <p>{cleaned_text || '(Empty text after preprocessing)'}</p>
        </div>
      </details>

      {/* Crisis Support Notice if Distress Detected */}
      {probability >= 0.60 && (
        <aside className="supportive-resources-box" role="complementary" aria-label="Support Resources">
          <HeartHandshake className="support-icon" size={24} aria-hidden="true" />
          <div className="support-body">
            <strong>Supportive Notice:</strong>
            <p>
              If you or someone you know is going through emotional distress or crisis, confidential and free help is available:
            </p>
            <ul className="support-links">
              <li><strong>US & Canada:</strong> Call or text <strong>988</strong> (Suicide & Crisis Lifeline)</li>
              <li><strong>Crisis Text Line:</strong> Text <strong>HOME</strong> to <strong>741741</strong></li>
              <li><strong>UK:</strong> Call <strong>111</strong> or text <strong>SHOUT</strong> to <strong>85258</strong></li>
              <li><strong>International:</strong> Visit <a href="https://findahelpline.com" target="_blank" rel="noopener noreferrer">findahelpline.com</a></li>
            </ul>
          </div>
        </aside>
      )}

      {/* Ethical Disclosure */}
      <footer className="result-footer">
        <p className="ethical-footer-note">{ethical_notice}</p>
      </footer>
    </article>
  );
}
