import csv
import pandas as pd
from pathlib import Path

def log_prediction(
    log_dir: Path,
    row_in: dict,
    y_hat: float,
    latency_ms: int,
    model_filename: str,
    model_version: str,
    target_name: str,
):
    """
    Enregistre une prédiction dans un fichier CSV de logs.

    Args:
        log_dir: dossier où écrire le fichier "predictions.csv"
        row_in: dictionnaire des features d'entrée
        y_hat: prédiction numérique
        latency_ms: durée d'inférence en millisecondes
        model_filename: nom du fichier modèle (ex: "model.joblib")
        model_version: version du modèle (issue du schéma)
        target_name: nom de la variable cible
    """
    log_file = log_dir / "predictions.csv"
    write_header = not log_file.exists()

    new_row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        **row_in,
        target_name: y_hat,
        "latency_ms": latency_ms,
        "model_file": model_filename,
        "model_version": model_version,
    }

    with log_file.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(new_row)
