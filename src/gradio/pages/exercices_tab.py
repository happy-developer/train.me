from typing import Union
from pathlib import Path

import gradio as gr
import pandas as pd

from ..helpers.exercices_tab_utilis import (
    DEFAULT_EXERCICES_PATH,
    _filter_by_level,
    _load_exercices,
)


def render_list_of_exercices(
    app_desc_ex: str,
    level_out: gr.Textbox,  # textbox du ML tab
    dataset_path: Union[str, Path] = DEFAULT_EXERCICES_PATH,
) -> None:
    """
    Onglet 'Exercices proposés' : tableau + panneau de détails.
    """

    df = _load_exercices(dataset_path)

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

    # Liste pour le panneau de détails
    has_name_col = "exercise_name" in df.columns
    exercice_choices = (
        sorted(df["exercise_name"].dropna().unique().tolist())
        if has_name_col
        else []
    )

    with gr.Tab("List of programs") as tab_ex:
        gr.Markdown(f"## {app_desc_ex}")

        # Niveau ML
        with gr.Row():
            level_display = gr.Textbox(
                label="Physical level",
                interactive=False,
                lines=1,
                max_lines=1,
            )

        # 🔍 Recherche
        search_box = gr.Textbox(
            label="Search in table",
            placeholder="Name, muscles, equipment, difficulty, source…",
        )

        # 🔽 Dropdown muscles — rempli dynamiquement
        muscle_filter = gr.Dropdown(
            label="Filter by target muscles",
            choices=["All"],
            value="All",
        )

        gr.Markdown(
            "The table below shows an overview of each exercise.\n\n"
            "- Use your ML level to adapt difficulty\n"
            "- Use the search box and muscle filter to refine\n"
            "- Select a program below to see full execution details\n"
        )

        # --- Tableau ---
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

        # ===== Callbacks =====

        # 1️⃣ Synchronisation + filtrage niveau à l'ouverture de l'onglet
        def _sync_on_tab_open(level_val: str):
            level_text = level_val or ""
            filtered = _filter_by_level(df_view, level_text)

            # muscles dynamiques selon niveau ML
            if "target_muscles" in filtered.columns:
                muscles = sorted(set(filtered["target_muscles"].dropna()))
            else:
                muscles = []

            return (
                level_text,
                filtered,
                gr.update(choices=["All"] + muscles, value="All"),
            )

        tab_ex.select(
            _sync_on_tab_open,
            inputs=[level_out],
            outputs=[level_display, table, muscle_filter],
        )

        # 2️⃣ Recherche texte + filtre muscle
        def _search_table(query: str, level_val: str, muscle_choice: str):
            base = _filter_by_level(df_view, level_val or "")

            # Filtre muscles
            if muscle_choice != "All" and "target_muscles" in base.columns:
                base = base[base["target_muscles"] == muscle_choice]

            # Recherche
            if query:
                df_str = base.astype(str)
                mask = df_str.apply(
                    lambda row: row.str.contains(query, case=False, na=False).any(),
                    axis=1,
                )
                base = base[mask]

            return base

        search_box.change(
            _search_table,
            inputs=[search_box, level_out, muscle_filter],
            outputs=table,
        )

        muscle_filter.change(
            _search_table,
            inputs=[search_box, level_out, muscle_filter],
            outputs=table,
        )
