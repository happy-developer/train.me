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
    encoder,
) -> pd.DataFrame:

    row = {}

    # === A) GENDER_1.0 ==========================================
    if "Gender_1.0" in expected_cols:
        g_str = ui_dict["Gender"]
        g_df = pd.DataFrame([[g_str]], columns=["Gender"])
        g_encoded = float(encoder.transform(g_df)[0, 0])
        row["Gender_1.0"] = 1.0 if g_encoded == 1.0 else 0.0

    # === B) WORKOUT_TYPE_* (HIIT / Strength / Yoga / Cardio) ===
    workout_types = ["Cardio", "Strength", "HIIT", "Yoga"]
    selected_wt = ui_dict["Workout_Type"]

    for wt in workout_types:
        col = f"Workout_Type_{wt}"
        if col in expected_cols:
            row[col] = 1.0 if selected_wt == wt else 0.0

    # # === C) BODY_PART_* (One-Hot) ===============================
    # body_parts = ["Abs", "Arms", "Back", "Chest", "Forearms", "Legs", "Shoulders"]
    # selected_bp = ui_dict["Body Part"]

    # for bp in body_parts:
    #     col = f"Body Part_{bp}"
    #     if col in expected_cols:
    #         row[col] = 1.0 if selected_bp == bp else 0.0


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
    encoder,
) -> Tuple[float, str]:
    """
    Implémentation officielle :
    UI → encodage Gender → scaling features → prédiction → inverse_transform cible.
    """

    # 0) Règle métier : si Workout_Type == "None" → prédiction forcée à 1 XP
    workout_type_raw = payload.get("Workout_Type")
    if isinstance(workout_type_raw, str) and workout_type_raw.strip().lower() == "none":
        y_xp = 1.0
        print(model_path)
        # Logging même si règle métier
        log_prediction(
            log_dir=log_dir,
            row_in=payload,
            y_hat=y_xp,
            latency_ms=0,
            model_filename=model_path.name,
            model_version=schema.get("model_version", "unknown"),
            target_name=target_name,
        )

        meta = (
            f"Model: {model_path.name} | "
            f"Version: {schema.get('model_version','?')} | "
            f"Features: {', '.join(internal_expected)} | "
            f"Rule applied: Workout_Type=None → y_xp=1"
        )
        return y_xp, meta

    # 1) Construire le DF interne
    X_raw = ui_to_internal_row(payload, internal_expected, encoder)

    # 2) Scaling des features
    X_scaled = pd.DataFrame(
        feature_scaler.transform(X_raw),
        columns=internal_expected,
        index=X_raw.index,
    )

    # 3) Prédiction standardisée
    y_std = float(model.predict(X_scaled)[0])

    # 4) Remise en unités réelles
    y_xp = float(target_scaler.inverse_transform(np.array([[y_std]]))[0, 0])
    y_xp = round(y_xp, 2)

    # 5) Logging
    log_prediction(
        log_dir=log_dir,
        row_in=payload,
        y_hat=y_xp,
        latency_ms=0,
        model_filename=model_path.name,
        model_version=schema.get("model_version", "unknown"),
        target_name=target_name,
    )

    meta = (
        f"Model: {model_path.name} | "
        f"Version: {schema.get('model_version','?')} | "
        f"Features: {', '.join(internal_expected)}"
    )
    return y_xp, meta

