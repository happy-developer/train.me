import json
import pandas as pd
from pathlib import Path


def read_model_report(path: Path) -> dict:
    """
    Lit le fichier JSON du rapport de modèle.
    Retourne un dictionnaire vide en cas d'erreur.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def report_summary_df(rep: dict) -> pd.DataFrame:
    """
    Construit un tableau récapitulatif des informations principales du rapport.
    """
    if not rep:
        return pd.DataFrame({
            "Info": [
                "created_at", "target", "n_features", "n_test_samples",
                "selected_model.type", "selected_model.class"
            ],
            "Valeur": ["-", "-", "-", "-", "-", "-"]
        })

    sel = rep.get("selected_model", {})
    rows = [
        ("created_at", rep.get("created_at", "-")),
        ("target", rep.get("target", "-")),
        ("n_features", rep.get("n_features", "-")),
        ("n_test_samples", rep.get("n_test_samples", "-")),
        ("selected_model.type", sel.get("model_type", "-")),
        ("selected_model.class", sel.get("model_class", "-")),
    ]
    return pd.DataFrame(rows, columns=["Informations", "Value"])


def report_metrics_df(rep: dict) -> pd.DataFrame:
    """
    Construit un tableau des métriques de performance par modèle.
    Colonnes : Modèle, MAE, RMSE, R2.
    """
    mets = rep.get("metrics_by_model", {})
    if not mets:
        return pd.DataFrame(columns=["Model", "MAE", "RMSE", "R2"])

    df = pd.DataFrame(mets).T.reset_index().rename(columns={"index": "Model"})

    # Conversion et arrondis
    for c in ("MAE", "RMSE", "R2"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").round(3)

    return df[["Model", "MAE", "RMSE", "R2"]]
