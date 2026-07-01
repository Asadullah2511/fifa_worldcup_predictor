import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_PATH, MODELS_DIR
from src.model_training import train_models
from src.predict import run_predictions
import joblib


def main():
    print("=" * 60)
    print("FIFA World Cup 2026 Predictor")
    print("=" * 60)

    print("\n[1/2] Training models...")
    stage_model, binary_models = train_models(TRAIN_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stage_model, MODELS_DIR / "stage_model.pkl")
    joblib.dump(binary_models, MODELS_DIR / "binary_models.pkl")
    print("Models saved.")

    print("\n[2/2] Generating 2026 predictions...")
    run_predictions()

    print("\n" + "=" * 60)
    print("Done! Check predictions/ directory for results.")
    print("=" * 60)


if __name__ == "__main__":
    main()
