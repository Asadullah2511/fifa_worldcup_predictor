from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "dataset"
MODELS_DIR = ROOT / "models"
PREDICTIONS_DIR = ROOT / "predictions"

TRAIN_PATH = DATASET_DIR / "train.csv"
TEST_PATH = DATASET_DIR / "test.csv"

TARGET_COLS = ["winner", "finalist", "semi_finalist", "quarter_finalist"]

NUM_FEATURES = [
    "goals_scored_last_4y", "goals_received_last_4y",
    "wins_last_4y", "losses_last_4y", "draws_last_4y",
    "world_cup_titles_before", "squad_total_market_value_eur",
    "fifa_rank_pre_tournament", "fifa_points_pre_tournament",
    "squad_avg_age", "world_cup_participations_before",
    "groups_passed_before", "round16_before",
    "quarterfinals_before", "semifinals_before", "finals_before",
]

CAT_FEATURES = ["continent"]

PASSTHROUGH_COLS = ["version", "team", "is_host"]
