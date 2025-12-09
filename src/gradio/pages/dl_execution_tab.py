from typing import Union
from pathlib import Path

import gradio as gr
import pandas as pd

from ..helpers.exercices_tab_utilis import (
    DEFAULT_EXERCICES_PATH,
    _load_exercices,
)
from ..generators.execution_generator import (
    build_execution_prompt,
    generate_execution_text,
    get_dl_execution_model_report_components,
)


def render_dl_execution_tab(
    app_desc_dl_exec: str,
    dataset_path: Union[str, Path] = DEFAULT_EXERCICES_PATH,
):
    """
    Onglet 'Deep Learning - Execution generator' :
    sélection d'un programme SANS execution pour générer son texte d'exécution.
    """

    df = _load_exercices(dataset_path)

    # ---- Filtrer uniquement les programmes sans execution ----
    if "execution" in df.columns:
        mask_no_exec = df["execution"].isna() | (df["execution"].astype(str).str.strip() == "")
        df_no_exec = df[mask_no_exec].copy()
    else:
        df_no_exec = df.copy()

    has_name_col = "exercise_name" in df_no_exec.columns
    exercice_choices = (
        sorted(df_no_exec["exercise_name"].dropna().unique().tolist())
        if has_name_col
        else []
    )

    # Colonnes affichées dans le tableau, y compris execution pour vérification
    base_cols = ["exercise_name", "target_muscles", "equipment", "difficulty", "execution"]
    selected_cols = [c for c in base_cols if c in df_no_exec.columns]

    with gr.Tab("Deep Learning - Execution generator") as tab_dl_exec:
        gr.Markdown(f"## {app_desc_dl_exec} - V3")

        gr.Markdown("### Program details")

        with gr.Row():
            exercice_selector = gr.Dropdown(
                label="Select a program without execution",
                choices=exercice_choices,
                value=exercice_choices[0] if exercice_choices else None,
            )

        details_md = gr.Markdown(
            value=(
                "Select a program **without execution description** "
                "to see its details."
            ),
        )

        # Tableau : programme sélectionné (avec colonne execution pour contrôle)
        selected_program_exec_df = gr.Dataframe(
            value=pd.DataFrame(columns=selected_cols),
            interactive=False,
            wrap=True,
            label="Selected program",
            row_count=(0, "dynamic"),
            col_count=(0, "dynamic"),
        )

        # Prompt auto-généré
        prompt_box = gr.Textbox(
            label="Execution prompt (auto-generated)",
            interactive=False,
            lines=3,
            max_lines=5,
        )

        # Bouton + sortie génération
        generate_btn = gr.Button("Generate execution")

        generated_exec = gr.Textbox(
            label="Generated execution",
            lines=10,
            max_lines=20,
        )

        gr.Markdown("### Execution generator – Model report")

        dl_summary = gr.Dataframe(
            value=pd.DataFrame({"Key": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Summary",
        )

        dl_model = gr.Dataframe(
            value=pd.DataFrame({"Key": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Model",
        )

        dl_training = gr.Dataframe(
            value=pd.DataFrame({"Key": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Training",
        )

        dl_metrics = gr.Dataframe(
            value=pd.DataFrame({"Metric": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Metrics",
        )

        # Callback de mise à jour details + tableau + prompt
        def _format_details_exec(ex_name: str):
            empty_df = pd.DataFrame(columns=selected_cols)
            empty_prompt = ""

            if not ex_name:
                return (
                    "Select a program **without execution description** "
                    "to see its details.",
                    empty_df,
                    empty_prompt,
                )

            subset = df_no_exec[df_no_exec["exercise_name"] == ex_name]
            if subset.empty:
                return "No details found for this program.", empty_df, empty_prompt

            row = subset.iloc[0]

            def get(col, default="—"):
                return row[col] if col in row and pd.notna(row[col]) else default

            parts = [
                f"**Name** : {get('exercise_name')}",
                f"**Target muscles** : {get('target_muscles')}",
                f"**Equipment** : {get('equipment')}",
                f"**Difficulty** : {get('difficulty')}",
                "",
                "This program currently has **no execution description**.",
                "You can generate a detailed execution using the Deep Learning model.",
            ]
            details_text = "\n".join(parts)

            sel_row = {c: get(c, "") for c in selected_cols}
            sel_df = pd.DataFrame([sel_row])

            # Prompt pour ce programme
            prompt = build_execution_prompt(row)

            return details_text, sel_df, prompt

        def _update_exec_report():
            df_summary, df_model_df, df_training_df, df_metrics_df = get_dl_execution_model_report_components()
            return df_summary, df_model_df, df_training_df, df_metrics_df



        exercice_selector.change(
            _format_details_exec,
            inputs=exercice_selector,
            outputs=[details_md, selected_program_exec_df, prompt_box],
        )

        # Génération à partir du prompt construit
        generate_btn.click(
            fn=generate_execution_text,
            inputs=prompt_box,
            outputs=generated_exec,
        )

        tab_dl_exec.select(
            _update_exec_report,
            inputs=None,  # ou [] mais None évite le warning
            outputs=[dl_summary, dl_model, dl_training, dl_metrics],
        )

    # On retourne le DF sélectionné pour l’execution generator
    return selected_program_exec_df
