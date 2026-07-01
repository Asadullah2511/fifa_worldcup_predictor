import numpy as np
import pandas as pd
import joblib

from src.config import MODELS_DIR, PREDICTIONS_DIR, TARGET_COLS
from src.data_preprocessing import (
    load_data,
    add_engineered_features,
    ALL_FEATURES,
    CAT_FEATURES,
)


N_TEAMS = {
    "winner": 1,
    "finalist": 2,
    "semi_finalist": 4,
    "quarter_finalist": 8,
}


def predict_2026(test_path, stage_pipe, binary_pipes):
    _, test_df = load_data(None, test_path)
    df = add_engineered_features(test_df)
    X = df[ALL_FEATURES + CAT_FEATURES]

    stage_probs = stage_pipe.predict_proba(X)
    stage_pred = stage_pipe.predict(X)

    binary_probs = {}
    for col in TARGET_COLS:
        proba = binary_pipes[col].predict_proba(X)
        binary_probs[col] = proba[:, 1]

    results = pd.DataFrame({"version": 2026, "team": df["team"]})

    for col in TARGET_COLS:
        results[f"{col}_binary_prob"] = binary_probs[col]

    results["stage_pred"] = stage_pred
    for stage in range(5):
        results[f"stage_{stage}_prob"] = stage_probs[:, stage]

    # Stage-derived probabilities for each target
    results["winner_ensemble_prob"] = stage_probs[:, 4]
    results["finalist_ensemble_prob"] = stage_probs[:, 3] + stage_probs[:, 4]
    results["semi_finalist_ensemble_prob"] = (
        stage_probs[:, 2] + stage_probs[:, 3] + stage_probs[:, 4]
    )
    results["quarter_finalist_ensemble_prob"] = (
        stage_probs[:, 1]
        + stage_probs[:, 2]
        + stage_probs[:, 3]
        + stage_probs[:, 4]
    )

    # Blend with binary model probabilities (average)
    for col in TARGET_COLS:
        results[f"{col}_blended_prob"] = (
            results[f"{col}_ensemble_prob"] + results[f"{col}_binary_prob"]
        ) / 2

    # Final prediction: take top N teams per target using blended probability
    for col in TARGET_COLS:
        n = N_TEAMS[col]
        prob_col = f"{col}_blended_prob"
        top_indices = results[prob_col].nlargest(n).index
        results[col] = 0
        results.loc[top_indices, col] = 1

    return results


def run_predictions():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    from src.config import TRAIN_PATH, TEST_PATH

    stage_pipe = joblib.load(MODELS_DIR / "stage_model.pkl")
    binary_pipes = joblib.load(MODELS_DIR / "binary_models.pkl")

    results = predict_2026(TEST_PATH, stage_pipe, binary_pipes)

    output_cols = ["version", "team"] + TARGET_COLS
    simple_output = results[output_cols].copy()
    simple_output.to_csv(PREDICTIONS_DIR / "predictions_2026.csv", index=False)

    results.to_csv(PREDICTIONS_DIR / "predictions_2026_detailed.csv", index=False)

    print("\n2026 World Cup Predictions:")
    print("=" * 60)
    for col in TARGET_COLS:
        predicted = results[results[col] == 1]["team"].tolist()
        print(f"\n{col.upper()}S ({len(predicted)}):")
        for t in predicted:
            prob = results.loc[results["team"] == t, f"{col}_blended_prob"].values[0]
            print(f"  {t:25s} (prob: {prob:.2%})")

    print(f"\n\nDetailed results saved to {PREDICTIONS_DIR / 'predictions_2026_detailed.csv'}")
    print(f"Summary saved to {PREDICTIONS_DIR / 'predictions_2026.csv'}")


if __name__ == "__main__":
    run_predictions()
