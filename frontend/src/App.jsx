import { useState } from "react";
import "./App.css";
import predictionData from "./all_predictions.json";

const STAGE_LABELS = {
  winner: "Champion",
  finalist: "Finalist",
  semi_finalist: "Semi-Finalist",
  quarter_finalist: "Quarter-Finalist",
};

function App() {
  const [selectedYear, setSelectedYear] = useState("2026");

  const { historical_winners, predictions } = predictionData;
  const currentPred = predictions[selectedYear];

  return (
    <div className="app">
      <header className="header">
        <div className="header-overlay"></div>
        <div className="header-content">
          <div className="trophy-icon">🏆</div>
          <h1>FiFa Predictor</h1>
          <p className="subtitle">
            AI-Powered World Cup Forecast Engine
          </p>
        </div>
      </header>

      <section className="section winners-section">
        <div className="container">
          <h2 className="section-title">
            <span className="title-accent"></span>
            Previous World Cup Winners
          </h2>
          <div className="winners-table-wrapper">
            <table className="winners-table">
              <thead>
                <tr>
                  <th>Year</th>
                  <th>Host</th>
                  <th>Winner</th>
                </tr>
              </thead>
              <tbody>
                {historical_winners.map((w) => (
                  <tr key={w.year}>
                    <td className="year-cell">{w.year}</td>
                    <td>{w.host}</td>
                    <td className="winner-cell">
                      <span className="winner-badge">{w.winner}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </section>

      <section className="section predictions-section">
        <div className="container">
          <h2 className="section-title">
            <span className="title-accent"></span>
            Future Tournament Predictions
          </h2>

          <div className="dropdown-wrapper">
            <label htmlFor="year-select">Select Tournament:</label>
            <div className="select-container">
              <select
                id="year-select"
                value={selectedYear}
                onChange={(e) => setSelectedYear(e.target.value)}
              >
                {Object.keys(predictions).map((y) => (
                  <option key={y} value={y}>
                    FIFA World Cup {y}
                  </option>
                ))}
              </select>
              <span className="select-arrow">▼</span>
            </div>
          </div>

          {currentPred && (
            <div className="predictions-grid">
              {["winner", "finalist", "semi_finalist", "quarter_finalist"].map(
                (stage) => {
                  const teams = currentPred.predictions[stage] || [];
                  return (
                    <div key={stage} className="prediction-card">
                      <div className={`card-header card-${stage}`}>
                        <span className="card-icon">
                          {stage === "winner"
                            ? "👑"
                            : stage === "finalist"
                              ? "🥈"
                              : stage === "semi_finalist"
                                ? "🏅"
                                : "⚽"}
                        </span>
                        <h3>{STAGE_LABELS[stage]}</h3>
                      </div>
                      <div className="card-body">
                        {teams.length === 0 ? (
                          <p className="no-prediction">No prediction</p>
                        ) : stage === "winner" ? (
                          <div className="winner-display">
                            <div className="winner-name">{teams[0].team}</div>
                            <div className="winner-progress">
                              <div
                                className="winner-progress-bar"
                                style={{ width: `${teams[0].probability}%` }}
                              ></div>
                            </div>
                            <div className="winner-prob">
                              {teams[0].probability.toFixed(1)}%
                            </div>
                          </div>
                        ) : (
                          <ul className="team-list">
                            {teams.map((t, i) => (
                              <li key={t.team} className="team-item">
                                <span className="team-rank">{i + 1}</span>
                                <span className="team-name">{t.team}</span>
                                <span className="team-prob">
                                  {t.probability.toFixed(1)}%
                                </span>
                              </li>
                            ))}
                          </ul>
                        )}
                      </div>
                    </div>
                  );
                },
              )}
            </div>
          )}
        </div>
      </section>

      <section className="section methodology-section">
        <div className="container">
          <h2 className="section-title">
            <span className="title-accent"></span>
            Methodology
          </h2>
          <div className="methodology-content">
            <div className="methodology-card">
              <div className="method-icon">📊</div>
              <h4>Historical Data</h4>
              <p>
                Trained on World Cup data from 2002-2022 with team performance
                metrics including goals, wins, FIFA rankings, and squad values.
              </p>
            </div>
            <div className="methodology-card">
              <div className="method-icon">🤖</div>
              <h4>XGBoost Model</h4>
              <p>
                Uses gradient-boosted decision trees with engineered features
                like goal difference, win rate, market value efficiency, and
                experience score.
              </p>
            </div>
            <div className="methodology-card">
              <div className="method-icon">📈</div>
              <h4>Ensemble Approach</h4>
              <p>
                Combines a multi-class stage classifier with binary classifiers
                for each knockout round, then blends probabilities for robust
                predictions.
              </p>
            </div>
          </div>
        </div>
      </section>

      <footer className="footer">
        <div className="container">
          <div className="disclaimer">
            <span className="disclaimer-icon">⚠️</span>
            <p>
              <strong>Disclaimer:</strong> This prediction is not 100%
              accurate. It is based on calculations of historical match
              performances, team statistics, and machine learning analysis.
              Football is unpredictable — that's what makes it beautiful.
            </p>
          </div>
          <p className="footer-text">
            FIFA World Cup Predictor &copy; {new Date().getFullYear()} &mdash;
            Built with data from 2002-2022 tournaments
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
