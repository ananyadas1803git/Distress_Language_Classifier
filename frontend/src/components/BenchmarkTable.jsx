import React from 'react';
import { BarChart3, HelpCircle, Trophy } from 'lucide-react';

export default function BenchmarkTable({ metrics }) {
  if (!metrics || metrics.length === 0) return null;

  return (
    <section className="card benchmark-card" aria-labelledby="benchmark-heading">
      <div className="section-header-row">
        <div>
          <h2 id="benchmark-heading" className="section-title">
            <BarChart3 size={20} aria-hidden="true" />
            <span>Baseline Models Comparative Evaluation</span>
          </h2>
          <p className="section-subtitle">
            Rigorous evaluation across 715 held-out test posts from the Dreaddit dataset.
          </p>
        </div>
      </div>

      <div className="table-responsive">
        <table className="benchmark-table">
          <thead>
            <tr>
              <th scope="col">Model Architecture</th>
              <th scope="col">Accuracy</th>
              <th scope="col">Precision</th>
              <th scope="col">Recall</th>
              <th scope="col">F1 Score</th>
              <th scope="col">ROC-AUC</th>
              <th scope="col">PR-AUC</th>
              <th scope="col">Brier Score</th>
              <th scope="col">ECE</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, idx) => {
              const isBestAcc = m.accuracy >= 0.73;
              const isBestRecall = m.recall >= 0.90;
              const isBestECE = m.ece <= 0.04;

              return (
                <tr key={idx} className={isBestAcc ? 'row-highlight' : ''}>
                  <th scope="row" className="model-col">
                    <strong>{m.model}</strong>
                    {isBestAcc && <span className="pill-badge pill-purple">Top Accuracy</span>}
                    {isBestRecall && <span className="pill-badge pill-green">High Recall</span>}
                    {isBestECE && <span className="pill-badge pill-blue">Best Calibration</span>}
                  </th>
                  <td className="num-col">{(m.accuracy * 100).toFixed(1)}%</td>
                  <td className="num-col">{(m.precision * 100).toFixed(1)}%</td>
                  <td className="num-col">{(m.recall * 100).toFixed(1)}%</td>
                  <td className="num-col"><strong>{(m.f1 * 100).toFixed(1)}%</strong></td>
                  <td className="num-col">{m.roc_auc.toFixed(3)}</td>
                  <td className="num-col">{m.pr_auc.toFixed(3)}</td>
                  <td className="num-col">{m.brier_score.toFixed(3)}</td>
                  <td className="num-col">{m.ece.toFixed(3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="metric-notes-grid">
        <div className="note-card">
          <h4>Expected Calibration Error (ECE)</h4>
          <p>
            Measures probability reliability by binning predictions. Lower ECE signifies that a predicted probability of 80% matches an actual 80% empirical incidence rate.
          </p>
        </div>
        <div className="note-card">
          <h4>Brier Score Loss</h4>
          <p>
            Mean squared difference between predicted probabilities and ground-truth binary outcomes (y in [0, 1]). A lower Brier score indicates superior calibration sharpness.
          </p>
        </div>
        <div className="note-card">
          <h4>Balanced Recall vs Precision</h4>
          <p>
            In distress language detection, false negatives risk missing acute distress signals, while false positives risk stigmatization. Evaluating both ensures ethical safety.
          </p>
        </div>
      </div>
    </section>
  );
}
