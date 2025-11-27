import os
from pathlib import Path
from typing import Union

import gradio as gr
import pandas as pd


# Le notebook s’exécute depuis son répertoire → on peut repartir du cwd
current_dir = Path.cwd()
json_path = current_dir / "src" / "notebooks" / "dataset_fusion" / "outputs"

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
    # Optionnel : on réindexe proprement
    df = df.reset_index(drop=True)
    return df


def render_list_of_exercices(
    app_desc_ex: str,
    dataset_path: Union[str, Path] = DEFAULT_EXERCICES_PATH,
) -> None:
    """
    Onglet 'Exercices proposés' : tableau filtrable/triable du dataset fusionné.
    """

    df = _load_exercices(dataset_path)

    # --- Vue "compacte" pour le tableau ---
    df_view = df.copy()
    if "execution" in df_view.columns:
        df_view["Execution (aperçu)"] = (
            df_view["execution"].astype(str).str.slice(0, 160).fillna("") + "..."
        )

        base_cols = [
            "exercise_name",
            "target_muscles",
            "equipment",
            "difficulty",
            "source_dataset",
        ]
        # On ne garde que les colonnes présentes
        cols = [c for c in base_cols if c in df_view.columns]
        cols.append("Execution (aperçu)")
        df_view = df_view[cols]

    columns = list(df_view.columns)

    # Colonne par défaut pour la recherche
    default_col = None
    for cand in ["exercise_name", "Exercise", "name", "Nom"]:
        if cand in columns:
            default_col = cand
            break
    if default_col is None:
        default_col = columns[0]

    with gr.Tab("Liste des programmes"):
        gr.Markdown(f"## {app_desc_ex}")
        gr.Markdown(
            "Le tableau ci-dessous affiche un **aperçu** de chaque exercice.\n\n"
            "- Clique sur les en-têtes pour trier\n"
            "- Utilise la recherche pour filtrer\n"
            "- La colonne *Execution (aperçu)* évite de prendre 100% de la largeur 😉"
        )

        with gr.Row():
            search_box = gr.Textbox(
                label="Recherche texte",
                placeholder="Nom, muscle, matériel…",
            )
            col_dropdown = gr.Dropdown(
                label="Colonne",
                choices=columns,
                value=default_col,
            )
            reset_btn = gr.Button("Réinitialiser")

        table = gr.Dataframe(
            value=df_view,
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
            col_count=(0, "dynamic"),
        )

        # --- Callbacks ---

        def _filter_table(query: str, col: str) -> pd.DataFrame:
            if not query:
                return df_view
            # On filtre sur la vue affichée (suffisant pour l’usage)
            mask = df_view[col].astype(str).str.contains(query, case=False, na=False)
            return df_view[mask]

        search_box.change(
            _filter_table,
            inputs=[search_box, col_dropdown],
            outputs=table,
        )
        col_dropdown.change(
            _filter_table,
            inputs=[search_box, col_dropdown],
            outputs=table,
        )

        def _reset():
            return "", df_view

        reset_btn.click(
            _reset,
            inputs=None,
            outputs=[search_box, table],
        )
