import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import TRAIN_PATH, MODELS_DIR
from src.model_training import train_models
from src.predict import run_predictions
import joblib


def main():
    print("=" * 60)
    print("FIFA World Cup Predictor - Full Pipeline")
    print("=" * 60)

    print("\n[1/2] Training models...")
    stage_model, binary_models = train_models(TRAIN_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stage_model, MODELS_DIR / "stage_model.pkl")
    joblib.dump(binary_models, MODELS_DIR / "binary_models.pkl")
    print("Models saved.")

    print("\n[2/2] Generating predictions (2026, 2030, 2034, 2038)...")
    run_predictions()
    subprocess.run(
        [sys.executable, str(ROOT / "generate_future_predictions.py")],
        check=True,
    )

    print("\n[3/3] Building React frontend...")
    frontend_dir = ROOT / "frontend"
    if frontend_dir.exists():
        subprocess.run(["npm", "install"], cwd=str(frontend_dir), check=True, capture_output=True)
        result = subprocess.run(
            ["npm", "run", "build"], cwd=str(frontend_dir), check=True, capture_output=True, text=True
        )
        print(result.stdout)
        print("Frontend built to frontend/dist/")

    print("\n" + "=" * 60)
    print("Done! Run the frontend with:")
    print("  cd frontend && npm run dev")
    print("Or open frontend/dist/index.html in your browser.")
    print("=" * 60)


if __name__ == "__main__":
    main()
