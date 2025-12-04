import os
from pathlib import Path
from typing import Union

import gradio as gr
import pandas as pd


# Le notebook s’exécute depuis son répertoire → on peut repartir du cwd
current_dir = Path.cwd()
json_path = current_dir / "src" / "gradio" / "data"

# Chemin par défaut vers ton JSON fusionné
DEFAULT_EXERCICES_PATH = Path(
    os.getenv(
        "DATASET_EXERCICES_FUSION",
        json_path / "dataset_exercices_fusion_20251127_2004.json",
    )
)

def _sync_on_tab_open(level_val: str) -> str:
    return level_val or ""

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


def render_list_of_exercices(
    app_desc_ex: str,
    level_out: gr.Textbox,  # 👈 nouveau param : textbox du ML tab
    dataset_path: Union[str, Path] = DEFAULT_EXERCICES_PATH,
) -> None:
    """
    Onglet 'Exercices proposés' : tableau + panneau de détails.
    """

    df = _load_exercices(dataset_path)
    _sync_level(level_out)
    # --- Vue "compacte" pour le tableau ---
    df_view = df.copy()
    if "execution" in df_view.columns:
        df_view["Execution (preview)"] = (
            df_view["execution"].astype(str).str.slice(0, 80).fillna("") + "..."
        )

        base_cols = [
            "exercise_name",
            "target_muscles",
            "equipment",
            "difficulty",
            "source_dataset",
        ]

        cols = [c for c in base_cols if c in df_view.columns]
        cols.append("Execution (preview)")
        df_view = df_view[cols]

    columns = list(df_view.columns)

    # Liste pour le panneau de détails
    has_name_col = "exercise_name" in df.columns
    exercice_choices = (
        sorted(df["exercise_name"].dropna().unique().tolist())
        if has_name_col else []
    )

    with gr.Tab("List of programs") as tab_ex:
        gr.Markdown(f"## {app_desc_ex}")

        # 🔹 Bloc d’affichage du niveau provenant du ML tab
        gr.Markdown("### Your physical level (from Machine Learning tab)")

        with gr.Row():
            level_display = gr.Textbox(
                label="Physical level",
                interactive=False,
                lines=1,
                max_lines=1,
            )

        gr.Markdown(
            "The table below shows an overview of each exercise.\n\n"
            "- Click on the headers to sort\n"
            "- Select a program below to see full execution details\n"
        )

        # --- Tableau compact ---
        table = gr.Dataframe(
            value=df_view,
            interactive=False,
            wrap=True,
            row_count=(0, "dynamic"),
            col_count=(0, "dynamic"),
        )

        # --- Panneau de détails ---
        if has_name_col:
            gr.Markdown("### Program details")

            with gr.Row():
                exercice_selector = gr.Dropdown(
                    label="Select a program",
                    choices=exercice_choices,
                    value=exercice_choices[0] if exercice_choices else None,
                )

            details_md = gr.Markdown(
                value="Select a program to see full description.",
            )

            def _format_details(ex_name: str) -> str:
                if not ex_name:
                    return "Select a program to see full description."

                subset = df[df["exercise_name"] == ex_name]
                if subset.empty:
                    return "No details found for this program."

                row = subset.iloc[0]

                def get(col, default="—"):
                    return row[col] if col in row and pd.notna(row[col]) else default

                parts = [
                    f"**Name** : {get('exercise_name')}",
                    f"**Target muscles** : {get('target_muscles')}",
                    f"**Equipment** : {get('equipment')}",
                    f"**Difficulty** : {get('difficulty')}",
                    f"**Source dataset** : {get('source_dataset')}",
                    "",
                ]

                exec_text = get("execution", "")
                if exec_text and exec_text != "—":
                    parts.append("**Execution** :")
                    parts.append("")
                    parts.append(exec_text)

                return "\n".join(parts)

            exercice_selector.change(
                _format_details,
                inputs=exercice_selector,
                outputs=details_md,
            )
     
            tab_ex.select(
                _sync_on_tab_open,
                inputs=[level_out],      # valeur provenant du ML tab
                outputs=[level_display], # textbox affichée dans l’onglet Exercices
            )
