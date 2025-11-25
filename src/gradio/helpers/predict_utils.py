import time
import pandas as pd
from pathlib import Path

from .log_utils import log_prediction
from .preprocess_utils import maybe_apply_feature_scaler, maybe_inverse_target
from ..model_loader import pipeline_has_scaler
from typing import Dict, List, Tuple
import numpy as np


def ui_to_internal_row(
    ui_dict: Dict[str, object],
    expected_cols: List[str],
    gender_encoder,
) -> pd.DataFrame:

    row = {}

    # Mapping ordinal pour Difficulty Level
    DIFF_LVL_MAP = {
        "Beginner": 0,
        "Intermediate": 1,
        "Advanced": 2,
    }

    for col in expected_cols:

        # --- 1) Gender_1.0 → binaire via gender_encoder ---
        if col == "Gender_1.0":
            g = ui_dict.get("Gender")
            g_df = pd.DataFrame([[g]], columns=["Gender"])
            g_encoded = float(gender_encoder.transform(g_df)[0, 0])
            row["Gender_1.0"] = 1.0 if g_encoded == 1.0 else 0.0
            continue

        # --- 2) Colonnes Workout_Type_* (One-Hot) ---
        if col.startswith("Workout_Type_"):
            # Exemple : Workout_Type_HIIT
            raw_type = ui_dict["Workout_Type"]
            category = col.replace("Workout_Type_", "")
            row[col] = 1.0 if category == raw_type else 0.0
            continue

        # --- 3) Difficulty Level (ordinal) ---
        if col == "Difficulty Level":
            diff_str = ui_dict.get("Difficulty Level")
            if diff_str not in DIFF_LVL_MAP:
                raise ValueError(f"Niveau de difficulté invalide: {diff_str}")
            row[col] = float(DIFF_LVL_MAP[diff_str])
            continue

        # --- 4) Toutes les autres colonnes (numériques) ---
        if col not in ui_dict:
            raise ValueError(f"Champ '{col}' manquant dans l'input UI.")

        row[col] = ui_dict[col]

    return pd.DataFrame([row], columns=expected_cols)


def predict_single(
    payload: Dict[str, object],
    internal_expected: List[str],
    model,
    feature_scaler,
    target_scaler,
    log_dir: Path,
    model_path: Path,
    schema: dict,
    target_name: str,
    gender_encoder,
) -> Tuple[float, str]:
    """
    Implémentation officielle :
    UI → encodage Gender → scaling features → prédiction → inverse_transform cible.
    """
    # 1) Construire le DF interne
    X_raw = ui_to_internal_row(payload, internal_expected, gender_encoder)

    # 2) Scaling des features
    X_scaled = pd.DataFrame(
        feature_scaler.transform(X_raw),
        columns=internal_expected,
        index=X_raw.index,
    )

    # 3) Prédiction standardisée
    y_std = float(model.predict(X_scaled)[0])

    # 4) Remise en unités réelles
    y_kcal = float(target_scaler.inverse_transform(np.array([[y_std]]))[0, 0])
    y_kcal = round(y_kcal, 2)

    # 5) Logging
    log_prediction(
        log_dir=log_dir,
        row_in=payload,
        y_hat=y_kcal,
        latency_ms=0,  # tu peux ajouter une mesure de temps si tu veux
        model_filename=model_path.name,
        model_version= schema.get("model_version", "unknown"),
        target_name=target_name,
    )

    meta = (
        f"Model: {model_path.name} | "
        f"Version: {schema.get('model_version','?')} | "
        f"Features: {', '.join(internal_expected)}"
    )
    return y_kcal, meta
