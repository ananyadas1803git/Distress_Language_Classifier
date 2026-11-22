import React from 'react';
import { ShieldAlert, Info, Sparkles, Scale } from 'lucide-react';

export default function Header() {
  return (
    <header className="app-header">
      <div className="header-top">
        <div className="logo-group">
          <div className="icon-wrapper">
            <Scale className="header-icon" aria-hidden="true" />
          </div>
          <div>
            <h1 className="header-title">Distress Language Classifier</h1>
            <p className="header-subtitle">
              Responsible NLP Framework with Calibrated Probabilities & Selective Abstention
            </p>
          </div>
        </div>
        <div className="header-badges">
          <span className="badge badge-purple">
            <Sparkles size={14} aria-hidden="true" /> 4 Baseline Models
          </span>
          <span className="badge badge-blue">
            <ShieldAlert size={14} aria-hidden="true" /> Uncertainty Handling
          </span>
        </div>
      </div>

      <aside className="ethical-banner" role="note" aria-label="Responsible AI Notice">
        <Info className="ethical-icon" size={20} aria-hidden="true" />
        <div className="ethical-content">
          <strong>Responsible AI & Ethical Mandate:</strong> This system identifies statistical linguistic associations 
          in anonymized online text. It provides calibrated probabilities and explicitly <em>abstains from deciding</em> when 
          uncertain. <strong>Predictions are strictly non-diagnostic and do not establish clinical psychological assessment.</strong>
        </div>
      </aside>
    </header>
  );
}
