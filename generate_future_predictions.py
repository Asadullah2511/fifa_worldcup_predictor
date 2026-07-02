import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import joblib
from copy import deepcopy

from src.config import MODELS_DIR, PREDICTIONS_DIR, TARGET_COLS
from src.data_preprocessing import add_engineered_features, ALL_FEATURES, CAT_FEATURES

N_TEAMS = {
    "winner": 1,
    "finalist": 2,
    "semi_finalist": 4,
    "quarter_finalist": 8,
}

FUTURE_YEARS = [2026, 2030, 2034, 2038]

np.random.seed(42)


def add_realistic_noise(df, year):
    df = df.copy()
    base_year = 2026
    decades = (year - base_year) // 4

    noise_scale = 0.05 + decades * 0.02

    noise_cols = [
        "goals_scored_last_4y", "goals_received_last_4y",
        "wins_last_4y", "losses_last_4y", "draws_last_4y",
        "fifa_rank_pre_tournament", "fifa_points_pre_tournament",
        "squad_total_market_value_eur", "squad_avg_age",
    ]
    for col in noise_cols:
        if col in df.columns:
            col_std = df[col].std() if df[col].std() > 0 else 1
            noise = np.random.normal(0, col_std * noise_scale, size=len(df))
            df[col] = df[col] + noise
            if col in ("fifa_rank_pre_tournament",):
                df[col] = df[col].clip(1, 211).astype(int)

    return df


def generate_future_test_data(base_df, year):
    df = base_df.copy()
    df["version"] = year
    df = add_realistic_noise(df, year)
    return df


def predict_year(test_df, stage_pipe, binary_pipes):
    df = add_engineered_features(test_df)
    X = df[ALL_FEATURES + CAT_FEATURES]

    stage_probs = stage_pipe.predict_proba(X)
    stage_pred = stage_pipe.predict(X)

    binary_probs = {}
    for col in TARGET_COLS:
        proba = binary_pipes[col].predict_proba(X)
        binary_probs[col] = proba[:, 1]

    year = df["version"].iloc[0]
    results = pd.DataFrame({"version": year, "team": df["team"]})

    for col in TARGET_COLS:
        results[f"{col}_binary_prob"] = binary_probs[col]

    results["stage_pred"] = stage_pred
    for stage in range(5):
        results[f"stage_{stage}_prob"] = stage_probs[:, stage]

    results["winner_ensemble_prob"] = stage_probs[:, 4]
    results["finalist_ensemble_prob"] = stage_probs[:, 3] + stage_probs[:, 4]
    results["semi_finalist_ensemble_prob"] = (
        stage_probs[:, 2] + stage_probs[:, 3] + stage_probs[:, 4]
    )
    results["quarter_finalist_ensemble_prob"] = (
        stage_probs[:, 1] + stage_probs[:, 2] + stage_probs[:, 3] + stage_probs[:, 4]
    )

    for col in TARGET_COLS:
        results[f"{col}_blended_prob"] = (
            results[f"{col}_ensemble_prob"] + results[f"{col}_binary_prob"]
        ) / 2

    for col in TARGET_COLS:
        n = N_TEAMS[col]
        prob_col = f"{col}_blended_prob"
        top_indices = results[prob_col].nlargest(n).index
        results[col] = 0
        results.loc[top_indices, col] = 1

    return results


def results_to_json(results_df):
    year = int(results_df["version"].iloc[0])
    output = {
        "year": year,
        "predictions": {},
    }
    for col in TARGET_COLS:
        predicted = results_df[results_df[col] == 1]
        teams = []
        for _, row in predicted.iterrows():
            teams.append({
                "team": str(row["team"]),
                "probability": round(float(row[f"{col}_blended_prob"]) * 100, 2),
                "stage_prob": round(float(row.get(f"stage_{['quarter_finalist', 'semi_finalist', 'finalist', 'winner'].index(col) if col != 'winner' else 4}_prob", 0)) * 100, 2) if col != 'quarter_finalist' else round(float(row.get("stage_1_prob", 0)) * 100, 2),
            })
        teams.sort(key=lambda x: x["probability"], reverse=True)
        output["predictions"][col] = teams
    return output


HISTORICAL_WINNERS = [
    {"year": 1930, "winner": "Uruguay", "host": "Uruguay"},
    {"year": 1934, "winner": "Italy", "host": "Italy"},
    {"year": 1938, "winner": "Italy", "host": "France"},
    {"year": 1950, "winner": "Uruguay", "host": "Brazil"},
    {"year": 1954, "winner": "West Germany", "host": "Switzerland"},
    {"year": 1958, "winner": "Brazil", "host": "Sweden"},
    {"year": 1962, "winner": "Brazil", "host": "Chile"},
    {"year": 1966, "winner": "England", "host": "England"},
    {"year": 1970, "winner": "Brazil", "host": "Mexico"},
    {"year": 1974, "winner": "West Germany", "host": "West Germany"},
    {"year": 1978, "winner": "Argentina", "host": "Argentina"},
    {"year": 1982, "winner": "Italy", "host": "Spain"},
    {"year": 1986, "winner": "Argentina", "host": "Mexico"},
    {"year": 1990, "winner": "West Germany", "host": "Italy"},
    {"year": 1994, "winner": "Brazil", "host": "United States"},
    {"year": 1998, "winner": "France", "host": "France"},
    {"year": 2002, "winner": "Brazil", "host": "South Korea/Japan"},
    {"year": 2006, "winner": "Italy", "host": "Germany"},
    {"year": 2010, "winner": "Spain", "host": "South Africa"},
    {"year": 2014, "winner": "Germany", "host": "Brazil"},
    {"year": 2018, "winner": "France", "host": "Russia"},
    {"year": 2022, "winner": "Argentina", "host": "Qatar"},
]


def main():
    print("=" * 60)
    print("FIFA World Cup - Future Predictions Generator")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading models...")
    stage_pipe = joblib.load(MODELS_DIR / "stage_model.pkl")
    binary_pipes = joblib.load(MODELS_DIR / "binary_models.pkl")

    base_test = pd.read_csv(ROOT / "dataset" / "test.csv")
    print(f"Loaded base test data ({len(base_test)} teams)")

    all_years_predictions = {}

    for year in FUTURE_YEARS:
        print(f"\n{'=' * 50}")
        print(f"Predicting {year} World Cup...")
        print(f"{'=' * 50}")

        if year == 2026:
            test_df = base_test.copy()
        else:
            test_df = generate_future_test_data(base_test, year)

        results = predict_year(test_df, stage_pipe, binary_pipes)
        json_data = results_to_json(results)
        all_years_predictions[str(year)] = json_data

        print(f"  Winner: {json_data['predictions']['winner'][0]['team']} ({json_data['predictions']['winner'][0]['probability']:.1f}%)")
        print(f"  Finalist: {json_data['predictions']['finalist'][0]['team']} ({json_data['predictions']['finalist'][0]['probability']:.1f}%)")

        json_path = PREDICTIONS_DIR / f"predictions_{year}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Saved to {json_path}")

    full_payload = {
        "historical_winners": HISTORICAL_WINNERS,
        "predictions": all_years_predictions,
    }

    with open(PREDICTIONS_DIR / "all_predictions.json", "w") as f:
        json.dump(full_payload, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"All predictions saved to {PREDICTIONS_DIR / 'all_predictions.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
