import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import NUM_FEATURES, CAT_FEATURES, TARGET_COLS


def load_data(train_path, test_path):
    train = pd.read_csv(train_path) if train_path else None
    test = pd.read_csv(test_path) if test_path else None
    return train, test


def derive_stage_target(df):
    stages = np.zeros(len(df), dtype=int)
    stages[df["winner"] == 1] = 4
    stages[(df["finalist"] == 1) & (df["winner"] == 0)] = 3
    stages[(df["semi_finalist"] == 1) & (df["finalist"] == 0)] = 2
    stages[(df["quarter_finalist"] == 1) & (df["semi_finalist"] == 0)] = 1
    return stages


def add_engineered_features(df):
    df = df.copy()
    df["goal_diff_last_4y"] = df["goals_scored_last_4y"] - df["goals_received_last_4y"]
    total_games = df["wins_last_4y"] + df["losses_last_4y"] + df["draws_last_4y"]
    df["win_rate_last_4y"] = np.where(total_games > 0, df["wins_last_4y"] / total_games, 0)
    df["avg_goals_scored"] = np.where(total_games > 0, df["goals_scored_last_4y"] / total_games, 0)
    df["avg_goals_received"] = np.where(total_games > 0, df["goals_received_last_4y"] / total_games, 0)
    df["market_value_per_rank"] = np.where(
        df["fifa_rank_pre_tournament"] > 0,
        df["squad_total_market_value_eur"] / df["fifa_rank_pre_tournament"],
        0,
    )
    df["experience_score"] = (
        df["world_cup_participations_before"]
        + df["groups_passed_before"]
        + df["round16_before"]
        + df["quarterfinals_before"]
        + df["semifinals_before"]
        + df["finals_before"]
    )
    return df


ENGINEERED_FEATURES = [
    "goal_diff_last_4y",
    "win_rate_last_4y",
    "avg_goals_scored",
    "avg_goals_received",
    "market_value_per_rank",
    "experience_score",
]

ALL_FEATURES = NUM_FEATURES + ENGINEERED_FEATURES


def build_preprocessor():
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, ALL_FEATURES),
            ("cat", categorical_transformer, CAT_FEATURES),
        ]
    )
    return preprocessor
