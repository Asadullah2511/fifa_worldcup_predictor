# 🏆 FiFa Predictor

> **AI-Powered World Cup Forecast Engine** — Predicts winners for the next three decades using XGBoost trained on 20+ years of tournament data.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3-EC1C24)
![React](https://img.shields.io/badge/React-19-61DAFB?logo=react)
![Vite](https://img.shields.io/badge/Vite-8-646CFF?logo=vite)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ⚡ What It Does

Feed the model team stats — goals, FIFA rank, market value, historical performance — and it predicts who lifts the trophy, who reaches the final, semi-finals, and quarter-finals.

**Currently supports predictions for:** `2026` · `2030` · `2034` · `2038`

---

## 🧠 How It Works

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────┐
│  Historical Data     │     │  XGBoost Ensemble    │     │  React Frontend  │
│  (2002–2022)         │ ──▶ │  Stage Classifier    │ ──▶ │  FIFA-themed UI  │
│  48 features/team    │     │  + 4 Binary Models   │     │  Dropdown + Cards│
└─────────────────────┘     └──────────────────────┘     └─────────────────┘
```

### Pipeline

| Step | What |
|------|------|
| **Data** | 193 team entries across 6 World Cups (2002–2022) |
| **Features** | Goals scored/received, wins/losses/draws, FIFA rank/points, market value, squad age, WC experience, continent, host advantage |
| **Engineering** | Goal diff, win rate, avg goals, market value per rank, experience score |
| **Model** | `XGBoost` multi-class (5 stages) + binary classifiers per knockout round, trained with Leave-One-Group-Out cross-validation |
| **Blending** | Stage probabilities + binary model probabilities averaged for final ranking |

### Model Performance

| Metric | Value |
|--------|-------|
| Cross-val Accuracy | ~65% |
| Cross-val F1 (weighted) | ~0.60 |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Node.js 20+
- A working venv

### Install

```bash
# Clone
git clone https://github.com/your-username/fifa-worldcup-predictor.git
cd fifa-worldcup-predictor

# Python deps
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux / macOS
pip install -r requirements.txt

# Frontend deps
cd frontend
npm install
cd ..
```

### Run Full Pipeline

```bash
.\venv\Scripts\python.exe run.py
```

This trains the model, generates predictions for all 4 tournaments, and builds the React frontend.

### Start the Frontend (Dev)

```bash
cd frontend
npm run dev
# Opens at http://localhost:5173
```

---

## 🖥️ Frontend Preview

| Section | Content |
|---------|---------|
| 🏆 **Header** | Animated trophy, green/gold gradient |
| 📜 **Previous Winners** | Full table of all 22 World Cup champions (1930–2022) |
| 🔮 **Predictions** | Dropdown to switch years, cards with Champion (progress bar), Finalists, Semi-Finalists, Quarter-Finalists |
| 📊 **Methodology** | Explains data source, XGBoost model, ensemble approach |
| ⚠️ **Disclaimer** | Predictions are not 100% accurate — football is unpredictable |

### Example: 2026 Winner Prediction

```
France — 58.4%  ████████████████████░░░░
```

---

## 📁 Project Structure

```
├── dataset/          # train.csv (2002-2022) + test.csv (2026)
├── src/
│   ├── config.py             # Paths & feature lists
│   ├── data_preprocessing.py # Feature engineering & pipeline
│   ├── model_training.py     # XGBoost training with LOGO CV
│   └── predict.py            # Prediction logic
├── models/           # Saved .pkl files
├── predictions/      # JSON + CSV output
├── frontend/         # React + Vite app
├── generate_future_predictions.py  # Multi-year prediction script
└── run.py            # Orchestrator
```

---

## ⚠️ Disclaimer

> **This prediction is not 100% accurate.** It is based on calculations of historical match performances, team statistics, and machine learning analysis. Football is unpredictable — that's what makes it beautiful.

---

## 📄 License

MIT — use it, tweak it, share it.
