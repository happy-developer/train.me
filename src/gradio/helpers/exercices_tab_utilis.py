from pathlib import Path
import os
import pandas as pd
from typing import Union



# Le fichier s’exécute depuis son répertoire → on peut repartir du cwd
current_dir = Path.cwd()
json_path = current_dir / "src" / "gradio" / "data"

# Chemin par défaut vers ton JSON fusionné
DEFAULT_EXERCICES_PATH = Path(
    os.getenv(
        "DATASET_EXERCICES_FUSION",
        json_path / "dataset_exercices_fusion_20251127_2004.json",
    )
)

def _load_exercices(path: Union[str, Path]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset exercices introuvable : {path}")

    df = pd.read_json(path)
    df = df.reset_index(drop=True)
    return df

def _sync_level(level_val: str) -> str:
    # Recopie la valeur du champ 'level_out' du ML tab
    return level_val or ""

def _filter_by_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Filtre automatiquement selon difficulty en fonction du niveau ML."""
    if "difficulty" not in df.columns:
        return df  # sécurité

    level = (level or "").strip().lower()

    if level == "beginner":
        allowed = ["beginner"]
    elif level == "intermediate":
        allowed = ["intermediate"]
    elif level == "advanced":
        allowed = ["advanced"]
    elif level == "expert":
        allowed = ["expert"]
    else:  # vide
        return df

    return df[df["difficulty"].str.lower().isin(allowed)]
