import os
from pathlib import Path
import yaml

# ---- Chaînes centralisées (sans logique) ----
# Dossiers / fichiers "métier"
MODEL_SUBDIR = ("models", "v1", "life_style_data")
MODEL_FILENAME = "model.joblib"
SCHEMA_FILENAME = "feature_schema.json"
REPORT_FILENAME = "model_report.json"
# Fichiers de scalers optionnels
FEATURE_SCALER_FILENAME = "feature_scaler.joblib"
TARGET_SCALER_FILENAME  = "target_scaler.joblib"
ENCODER_FILENAME = "encoder.joblib"

# Le notebook s’exécute depuis son répertoire → on peut repartir du cwd
current_dir = Path.cwd()
# Valeurs par défaut UI (non normalisées)
config_path = Path(current_dir / "src/config/ui_defaults.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    UI_DEFAULTS = yaml.safe_load(f)["UI_DEFAULTS"]

# Exemples UI (affichés sous les sliders)
examples_path = Path(current_dir / "src/config/ui_examples.yaml")
with open(examples_path, "r", encoding="utf-8") as f:
    UI_EXAMPLES = yaml.safe_load(f)["UI_EXAMPLES"]



# Base de validation (relative au repo src/)
DB_RELATIVE = ("data", "processed", "life_style_data", "life_style_data_val.db")

# ---- Résolution des chemins (avec ENV overrides optionnels) ----
def build_paths(src_dir: Path) -> dict[str, Path]:
    """
    Construit tous les chemins nécessaires à l'app Gradio à partir de src_dir.
    Les variables d'environnement suivantes peuvent override :
      - MODEL_PATH, SCHEMA_PATH, MODEL_REPORT_PATH
    """
    model_dir = src_dir.joinpath(*MODEL_SUBDIR)

    model_path_env = os.getenv("MODEL_PATH")
    schema_path_env = os.getenv("SCHEMA_PATH")
    report_path_env = os.getenv("MODEL_REPORT_PATH")

    paths = {
        "MODEL_DIR": model_dir,
        "MODEL_PATH": Path(model_path_env) if model_path_env else model_dir / MODEL_FILENAME,
        "SCHEMA_PATH": Path(schema_path_env) if schema_path_env else model_dir / SCHEMA_FILENAME,
        "LOGS_DIR": src_dir / "logs",
        "DB_PATH": src_dir.joinpath(*DB_RELATIVE),
        "REPORT_PATH": Path(report_path_env) if report_path_env else model_dir / REPORT_FILENAME,
        "FEATURE_SCALER_PATH": model_dir / FEATURE_SCALER_FILENAME,
        "TARGET_SCALER_PATH":  model_dir / TARGET_SCALER_FILENAME,
        "ENCODER_PATH": model_dir / ENCODER_FILENAME,
    }
    return paths
