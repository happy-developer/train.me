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
    """
    Transforme un dict UI {Age, Weight (kg), ..., Workout_Type}
    en DataFrame 1 ligne avec colonnes internes alignées sur expected_cols.
    """
    row = {}

    for col in expected_cols:

        # -------------------------
        # 1) Recontruction Gender_1.0
        # -------------------------
        if col == "Gender_1.0":
            if "Gender" not in ui_dict:
                raise ValueError("Champ 'Gender' manquant dans l'input UI.")

            g_str = ui_dict["Gender"]
            if g_str not in ("Male", "Female"):
                raise ValueError("Genre invalide. Valeurs autorisées : Male / Female.")

            g_df = pd.DataFrame([[g_str]], columns=["Gender"])
            g_encoded = float(gender_encoder.transform(g_df)[0, 0])

            row["Gender_1.0"] = 1.0 if g_encoded == 1.0 else 0.0

        # -------------------------
        # 2) Reconstruction Workout_Type_* (HIIT / Strength / Yoga)
        # -------------------------
        elif col.startswith("Workout_Type_"):
            if "Workout_Type" not in ui_dict:
                raise ValueError(
                    "Champ 'Workout_Type' manquant dans l'input UI "
                    f"alors que la colonne '{col}' est attendue."
                )

            workout_type = ui_dict["Workout_Type"]
            suffix = col.split("Workout_Type_", 1)[1]  # ex: "HIIT"

            # Cardio = toutes les colonnes = 0.0
            row[col] = 1.0 if workout_type == suffix else 0.0

        # -------------------------
        # 3) Toutes les autres colonnes (numériques brutes)
        # -------------------------
        else:
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
