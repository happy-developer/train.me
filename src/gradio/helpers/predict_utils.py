import time
import pandas as pd
from pathlib import Path

from .log_utils import log_prediction
from .preprocess_utils import maybe_apply_feature_scaler, maybe_inverse_target
from ..model_loader import pipeline_has_scaler


def predict_single(
    payload: dict,
    model,
    expected_order: list[str],
    log_dir: Path,
    model_path: Path,
    schema: dict,
    target_name: str,
    feature_scaler=None,   # <-- nouveau
    target_scaler=None,    # <-- nouveau
):
    t0 = time.time()

    # 1) Vérif & construction X dans l'ordre
    missing = [c for c in expected_order if c not in payload]
    if missing:
        raise ValueError(f"Champs manquants dans l'input UI : {missing}")

    X_raw = pd.DataFrame([[payload[c] for c in expected_order]], columns=expected_order)
    X_raw = X_raw.apply(pd.to_numeric, errors="raise")

    # 2) Déterminer si le modèle intègre déjà un scaler (Pipeline)
    uses_internal_scaling = pipeline_has_scaler(model)

    # 3) Appliquer le scaling uniquement si nécessaire
    if uses_internal_scaling:
        X_for_pred = X_raw
        scaler_flag = "internal"
    else:
        if feature_scaler is None:
            raise RuntimeError(
                "Le modèle n'intègre pas de scaler et aucun feature_scaler.joblib n'a été trouvé."
            )
        X_for_pred = maybe_apply_feature_scaler(X_raw, feature_scaler)
        scaler_flag = "external"

    # 4) Prédiction
    y_pred = float(model.predict(X_for_pred).squeeze())

    # 5) Remise en unités réelles de la cible si normalisée
    y_final, inverted = maybe_inverse_target(y_pred, target_scaler)

    latency_ms = int((time.time() - t0) * 1000)

    # 6) Logging
    log_prediction(
        log_dir=log_dir,
        row_in={k: payload[k] for k in expected_order},
        y_hat=round(y_final, 2),
        latency_ms=latency_ms,
        model_filename=model_path.name,
        model_version=schema.get("model_version", "unknown"),
        target_name=target_name,
    )

    # 7) Meta lisible
    meta = (
        f"Latency: {latency_ms} ms | "
        f"Model: {model_path.name} | Version: {schema.get('model_version','?')} | "
        f"Scaling: {scaler_flag}{' + target_inverse' if inverted else ''}"
    )
    return round(y_final, 2), meta
