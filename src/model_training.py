import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import MODELS_DIR, TARGET_COLS
from src.data_preprocessing import (
    load_data,
    add_engineered_features,
    derive_stage_target,
    build_preprocessor,
    ALL_FEATURES,
    CAT_FEATURES,
)


def prepare_training_data(train_df):
    df = add_engineered_features(train_df)
    y_stage = derive_stage_target(df)
    y_multi = df[TARGET_COLS].values
    X = df[ALL_FEATURES + CAT_FEATURES]
    groups = df["version"].values
    return X, y_stage, y_multi, groups


def train_stage_model(X, y, preprocessor, groups):
    class_counts = np.bincount(y)
    scale_pos_weight = max(class_counts) / class_counts
    sample_weights = np.array([scale_pos_weight[c] for c in y])

    base_params = {
        "objective": "multi:softprob",
        "num_class": 5,
        "n_estimators": 1000,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "gamma": 0.5,
        "min_child_weight": 5,
        "random_state": 42,
        "early_stopping_rounds": 50,
    }

    X_processed = preprocessor.fit_transform(X)

    logo = LeaveOneGroupOut()
    cv_scores = []
    best_n_estimators = []

    for fold, (train_idx, val_idx) in enumerate(logo.split(X_processed, y, groups)):
        X_tr, X_val = X_processed[train_idx], X_processed[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        sw_tr = sample_weights[train_idx]

        model = XGBClassifier(**base_params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sw_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        y_pred = model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, average="weighted")
        cv_scores.append((acc, f1))
        best_n_estimators.append(model.best_iteration + 1)

        val_years = np.unique(groups[val_idx])
        print(f"  Val year {val_years[0]}: acc={acc:.4f}, f1={f1:.4f}")

    print(f"\nLOGO CV Accuracy: {np.mean([s[0] for s in cv_scores]):.4f} +/- {np.std([s[0] for s in cv_scores]):.4f}")
    print(f"LOGO CV F1 (weighted): {np.mean([s[1] for s in cv_scores]):.4f}")

    best_n = int(np.median(best_n_estimators)) + 100
    final_params = {k: v for k, v in base_params.items() if k != "early_stopping_rounds"}
    final_params["n_estimators"] = best_n

    final_model = XGBClassifier(**final_params)
    final_model.fit(X_processed, y, sample_weight=sample_weights)

    full_pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", final_model),
    ])

    return full_pipe


def train_individual_binary_models(X, y_multi, preprocessor, groups):
    X_processed = preprocessor.fit_transform(X)
    binary_pipes = {}

    class_weights = {}
    for i, col in enumerate(TARGET_COLS):
        pos = int(y_multi[:, i].sum())
        neg = len(y_multi) - pos
        class_weights[col] = neg / pos if pos > 0 else 1

    base_params = {
        "objective": "binary:logistic",
        "n_estimators": 800,
        "max_depth": 3,
        "learning_rate": 0.03,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.5,
        "reg_lambda": 2.0,
        "random_state": 42,
        "early_stopping_rounds": 30,
    }

    logo = LeaveOneGroupOut()

    for i, col in enumerate(TARGET_COLS):
        y_col = y_multi[:, i]
        scale = class_weights[col]
        params = base_params.copy()
        params["scale_pos_weight"] = scale

        best_n_list = []

        for fold, (train_idx, val_idx) in enumerate(logo.split(X_processed, y_col, groups)):
            X_tr, X_val = X_processed[train_idx], X_processed[val_idx]
            y_tr, y_val = y_col[train_idx], y_col[val_idx]

            model = XGBClassifier(**params)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            best_n_list.append(model.best_iteration + 1)

        best_n = int(np.median(best_n_list)) + 50
        final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        final_params["n_estimators"] = best_n

        final_model = XGBClassifier(**final_params)
        final_model.fit(X_processed, y_col)

        binary_pipes[col] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", final_model),
        ])

    return binary_pipes


def train_models(train_path):
    train_df, _ = load_data(train_path, None)
    X, y_stage, y_multi, groups = prepare_training_data(train_df)

    print("=" * 60)
    print("Training Stage Prediction Model (Multi-Class)")
    print("=" * 60)
    preprocessor = build_preprocessor()
    stage_pipe = train_stage_model(X, y_stage, preprocessor, groups)

    print("\n" + "=" * 60)
    print("Training Individual Binary Models")
    print("=" * 60)
    preprocessor2 = build_preprocessor()
    binary_pipes = train_individual_binary_models(X, y_multi, preprocessor2, groups)

    return stage_pipe, binary_pipes


if __name__ == "__main__":
    from src.config import TRAIN_PATH
    stage_model, binary_models = train_models(TRAIN_PATH)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(stage_model, MODELS_DIR / "stage_model.pkl")
    joblib.dump(binary_models, MODELS_DIR / "binary_models.pkl")
    print("\nModels saved to", MODELS_DIR)
