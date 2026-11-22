import React from 'react';
import { Cpu } from 'lucide-react';

export default function ModelSelector({ models, selectedModel, onSelectModel, disabled }) {
  return (
    <fieldset className="model-selector-fieldset" disabled={disabled}>
      <legend className="fieldset-legend">
        <Cpu size={16} aria-hidden="true" />
        <span>Select Baseline Classifier (4 Available)</span>
      </legend>

      <div className="models-grid" role="radiogroup" aria-label="Classifier selection">
        {models.map((m) => {
          const isSelected = selectedModel === m.name;
          const inputId = `model-radio-${m.id}`;

          return (
            <div
              key={m.id}
              className={`model-option-card ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectModel(m.name)}
            >
              <div className="model-card-header">
                <input
                  type="radio"
                  id={inputId}
                  name="classifier_model"
                  value={m.name}
                  checked={isSelected}
                  onChange={() => onSelectModel(m.name)}
                  className="model-radio-input"
                />
                <label htmlFor={inputId} className="model-name-label">
                  {m.name.replace('TF-IDF + ', '')}
                </label>
                {m.is_calibrated && (
                  <span className="calibrated-tag" title="Calibrated probability outputs">
                    Calibrated
                  </span>
                )}
              </div>

              <p className="model-algorithm">{m.algorithm}</p>
              <p className="model-strengths"><strong>Strength:</strong> {m.strengths}</p>
            </div>
          );
        })}
      </div>
    </fieldset>
  );
}
