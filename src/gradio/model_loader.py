import json
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

def load_model_and_schema(model_path: Path, schema_path: Path):
    """
    Charge le modèle .joblib et le schéma JSON associé.
    Retourne (model, schema, target_name, features, expected_order).
    """
    # --- Chargement du modèle ---
    print(model_path)
    model = joblib.load(model_path)

    # --- Chargement du schéma ---
    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    # --- Extraction des infos principales ---
    target_name = schema.get("target", "Experience_level")
    features = schema.get("features", [])

    # --- Ordre des colonnes attendu par le modèle ---
    if hasattr(model, "feature_names_in_"):
        expected_order = list(model.feature_names_in_)
    else:
        expected_order = [f["name"] for f in features]

    return model, schema, target_name, features, expected_order

def load_optional_joblib(path: Path):
    try:
        if path and Path(path).exists():
            return joblib.load(path)
    except Exception:
        pass
    return None

def pipeline_has_scaler(p) -> bool:
    """
    True si 'p' est un Pipeline sklearn qui contient un scaler connu.
    """
    if not isinstance(p, Pipeline):
        return False
    scaler_types = (StandardScaler, MinMaxScaler, RobustScaler)
    return any(isinstance(step, scaler_types) for _, step in p.named_steps.items())