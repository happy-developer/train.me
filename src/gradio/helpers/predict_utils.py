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

    # === A) GENDER_1.0 ==========================================
    if "Gender_1.0" in expected_cols:
        g_str = ui_dict["Gender"]
        g_df = pd.DataFrame([[g_str]], columns=["Gender"])
        g_encoded = float(gender_encoder.transform(g_df)[0, 0])
        row["Gender_1.0"] = 1.0 if g_encoded == 1.0 else 0.0

    # === B) WORKOUT_TYPE_* (HIIT / Strength / Yoga / Cardio) ===
    workout_types = ["Cardio", "Strength", "HIIT", "Yoga"]
    selected_wt = ui_dict["Workout_Type"]

    for wt in workout_types:
        col = f"Workout_Type_{wt}"
        if col in expected_cols:
            row[col] = 1.0 if selected_wt == wt else 0.0

    # === C) BODY_PART_* (One-Hot) ===============================
    body_parts = ["Abs", "Arms", "Back", "Chest", "Forearms", "Legs", "Shoulders"]
    selected_bp = ui_dict["Body Part"]

    for bp in body_parts:
        col = f"Body Part_{bp}"
        if col in expected_cols:
            row[col] = 1.0 if selected_bp == bp else 0.0

    # === D) DIFFICULTY LEVEL (Ordinal) ==========================
    DIFF_LVL_MAP = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}

    if "Difficulty Level" in expected_cols:
        lvl_str = ui_dict["Difficulty Level"]
        row["Difficulty Level"] = float(DIFF_LVL_MAP[lvl_str])

    # === E) COPIE DIRECTE DES AUTRES COLONNES ===================
    for col in expected_cols:
        if col in row:
            continue
        if col in ui_dict:
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
