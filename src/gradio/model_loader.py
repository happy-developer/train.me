import json
import joblib
from pathlib import Path

def load_model_and_schema(model_path: Path, schema_path: Path):
    """
    Charge le modèle .joblib et le schéma JSON associé.
    Retourne (model, schema, target_name, features, expected_order).
    """
    # --- Chargement du modèle ---
    model = joblib.load(model_path)

    # --- Chargement du schéma ---
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # --- Extraction des infos principales ---
    target_name = schema.get("target", "Calories_Burned")
    features = schema.get("features", [])

    # --- Ordre des colonnes attendu par le modèle ---
    if hasattr(model, "feature_names_in_"):
        expected_order = list(model.feature_names_in_)
    else:
        expected_order = [f["name"] for f in features]

    return model, schema, target_name, features, expected_order