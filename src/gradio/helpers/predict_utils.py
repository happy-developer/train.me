import time
import pandas as pd
from pathlib import Path
from .log_utils import log_prediction


def predict_single(
    payload: dict,
    model,
    expected_order: list[str],
    log_dir: Path,
    model_path: Path,
    schema: dict,
    target_name: str,
):
    """
    Effectue une prédiction unique sur la base du payload fourni.

    Args:
        payload: dictionnaire des entrées utilisateur
        model: modèle scikit-learn chargé
        expected_order: ordre des features attendu par le modèle
        log_dir: dossier de logs
        model_path: chemin complet du modèle
        schema: schéma JSON (pour version, target, etc.)
        target_name: nom de la variable cible

    Returns:
        tuple (y_pred: float, meta: str)
    """
    t0 = time.time()

    # --- Vérification des champs ---
    missing = [c for c in expected_order if c not in payload]
    if missing:
        raise ValueError(f"Champs manquants dans l'input UI : {missing}")

    # --- Construction du DataFrame dans l'ordre attendu ---
    X_one = pd.DataFrame([[payload[c] for c in expected_order]], columns=expected_order)
    X_one = X_one.apply(pd.to_numeric, errors="raise")

    # --- Prédiction ---
    y_pred = float(model.predict(X_one).squeeze())
    y_pred = round(y_pred, 2)
    latency_ms = int((time.time() - t0) * 1000)

    # --- Logging ---
    log_prediction(
        log_dir=log_dir,
        row_in={k: payload[k] for k in expected_order},
        y_hat=y_pred,
        latency_ms=latency_ms,
        model_filename=model_path.name,
        model_version=schema.get("model_version", "unknown"),
        target_name=target_name,
    )

    # --- Métadonnées ---
    meta = (
        f"Latency: {latency_ms} ms | "
        f"Model: {model_path.name} | "
        f"Version: {schema.get('model_version','?')}"
    )

    return y_pred, meta
