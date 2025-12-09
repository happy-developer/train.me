from pathlib import Path
import gradio as gr
import pandas as pd

from ..helpers.schema_utils import get_bounds
from ..helpers.sqlite_utils import load_val_subset
from ..helpers.predict_utils import predict_single
from ..helpers.report_utils import (
    read_model_report, report_summary_df, report_metrics_df
)
from typing import Dict, List


def render_ml_tab(
    app_desc_ml: str,
    feature_specs: List[dict],
    ui_feature_names: List[str],
    internal_expected: List[str],
    target_name: str,
    schema: dict,
    ui_examples: List[Dict],
    db_path: Path,
    model,
    logs_dir: Path,
    model_path: Path,
    feature_scaler,
    target_scaler,
    encoder,
    report_path: Path,
    on_load=None,
) -> None:
    display_headers = [target_name] + [c for c in ui_feature_names if c != target_name]

    with gr.Tab(f"Machine Learning - {app_desc_ml} ({schema.get('model_name','model')})"):
        # ====== Ligne principale (inputs/pred) ======
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Assess your physical level")
                comps, names = [], []

                wf_comp = None  # Workout_Frequency (days/week)
                wt_comp = None  # Workout_Type

                for spec in feature_specs:
                    name = spec["name"]
                    if name == "Experience_Level":
                        comp = gr.Slider(0, 3, value=0, step=0, label=name, visible=False)

                        comps.append(comp)
                        names.append(name)
                        continue

                    if name == "Gender":
                        # UI = Radio pour Male / Female
                        choices = spec.get("enum", ["Male", "Female"])
                        comp = gr.Radio(
                            choices=choices,
                            value=choices[0],
                            label="Gender",
                        )

                    elif name == "Workout_Type":
                        # UI = Liste déroulante pour le type de séance
                        choices = spec.get("enum", ["Cardio", "Strength", "HIIT", "Yoga"])
                        comp = gr.Dropdown(
                            choices=choices,
                            value=choices[0],
                            label="Workout_Type",
                        )
                        wt_comp = comp  # on garde une référence

                    else:
                        vmin, vmax, default, step = get_bounds(spec, schema)
                        comp = gr.Slider(
                            vmin,
                            vmax,
                            value=default,
                            step=step,
                            label=name,
                        )

                        if name == "Workout_Frequency (days/week)":
                            wf_comp = comp  # on garde une référence

                    comps.append(comp)
                    names.append(name)

                btn = gr.Button("Predict", variant="primary")

            with gr.Column():
                gr.Markdown("### Calculs")

                bmi_out = gr.Number(
                    label="BMI (Body Mass Index)",
                    interactive=False,
                    precision=2,
                )
                fat_out = gr.Number(
                    label="Body fat percentage (%)",
                    interactive=False,
                    precision=2,
                )

                gr.Markdown("---")

                gr.Markdown("### Prédiction")

                y_out = gr.Number(
                    label="Physical experience level notation",
                    interactive=False,
                    precision=2,
                )

                # Nouveau champ texte interprétation du niveau
                level_out = gr.Textbox(
                    label="Physical level (text)",
                    interactive=False,
                    lines=1,
                    max_lines=1,
                )

                meta_out = gr.Textbox(
                    label="Informations",
                    interactive=False,
                    lines=4,
                    max_lines=8,
                )

        # ====== Exemples ======
        examples_dicts = ui_examples or [schema.get("example_payload", {})]

        # --- Tableau complet (pour l'affichage) ---
        df_examples_full = pd.DataFrame(examples_dicts)

        # On impose Experience_Level en première colonne si elle existe
        if "Experience_Level" in df_examples_full.columns:
            ordered_cols = (
                ["Experience_Level"] +
                [c for c in df_examples_full.columns if c != "Experience_Level"]
            )
            df_examples_full = df_examples_full[ordered_cols]

        # --- Exemples envoyés au modèle (uniquement les features d’entrée) ---
        df_examples_for_gradio = df_examples_full[names]  # on garde seulement les features utiles
        rows = df_examples_for_gradio.values.tolist()

        gr.Examples(
            examples=rows,
            inputs=comps,
            label="Exemples (sélection rapide)"
        )

        gr.Markdown("---")

        # ====== Prédiction ======

        def _interpret_level(y_val) -> str:
            """Map numeric prediction to textual level."""
            try:
                if y_val is None:
                    return ""
                v = float(y_val)
            except Exception:
                return ""

            if 0 <= v < 1:
                return "Beginner"
            elif 1 <= v < 2:
                return "Intermediate"
            elif 2 <= v < 2.5:
                return "Advanced"
            elif 2.5 <= v <= 3:
                return "Expert"
            return ""
        
        def xp_to_label_safe(xp: float) -> str:
            try:
                x = float(xp)
            except (TypeError, ValueError):
                return "Unknown"

            if x < 2:
                return "Beginner"
            elif x < 4:
                return "Intermediate"
            elif x < 6:
                return "Advanced"
            else:
                return "Athlete"

        def _fn(*vals):
            payload = {k: v for k, v in zip(names, vals)}

            # Gradio peut renvoyer les valeurs numériques en str → on force
            if "Age" in payload:
                payload["Age"] = float(payload["Age"])
            if "Weight (kg)" in payload:
                payload["Weight (kg)"] = float(payload["Weight (kg)"])
            if "Height (m)" in payload:
                payload["Height (m)"] = float(payload["Height (m)"])
            if "Workout_Frequency (days/week)" in payload:
                payload["Workout_Frequency (days/week)"] = float(
                    payload["Workout_Frequency (days/week)"]
                )

            y_xp, meta = predict_single(
                payload=payload,
                internal_expected=internal_expected,
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                log_dir=logs_dir,
                model_path=model_path,
                schema=schema,
                target_name=target_name,
                encoder=encoder,
            )

            print("[DEBUG] y_xp =", y_xp)
            print("[DEBUG] meta =", meta)

            level_text = xp_to_label_safe(y_xp)
            info_text = str(meta)

            print("[DEBUG] level_text =", level_text)
            print("[DEBUG] info_text  =", info_text)


            # 2) Calcul BMI & Body Fat %
            bmi = None
            fat_pct = None
            try:
                w = float(payload.get("Weight (kg)", 0) or 0)
                h = float(payload.get("Height (m)", 0) or 0)
                if h > 0:
                    bmi = round(w / (h ** 2), 2)

                age_val = payload.get("Age")
                gender_raw = str(payload.get("Gender", "")).lower()

                if bmi is not None and age_val not in (None, ""):
                    age_f = float(age_val)
                    # 1 = homme, 0 = femme (IMG formule classique)
                    sex_flag = 1.0 if gender_raw.startswith("m") else 0.0

                    fat_pct = 1.20 * bmi + 0.23 * age_f - 10.8 * sex_flag - 5.4
                    fat_pct = round(fat_pct, 2)
            except Exception:
                bmi = None
                fat_pct = None

            # 4) Retourner les 5 sorties Gradio
            return y_xp, level_text, bmi, fat_pct, meta

        btn.click(_fn, comps, [y_out, level_out, bmi_out, fat_out, meta_out])

        gr.Markdown("---")

        # ====== Rapport modèle ======
        rep = read_model_report(report_path)
        df_sum = report_summary_df(rep)
        df_mets = report_metrics_df(rep)

        gr.Markdown("### Machine Learning model evaluation report")
        gr.Dataframe(
            value=df_sum,
            interactive=False,
            wrap=True,
            label="Summary",
            row_count=(0, "dynamic"),
            col_count=df_sum.shape[1],
        )
        gr.Dataframe(
            value=df_mets,
            interactive=False,
            wrap=True,
            label="Metrics by model",
            row_count=(0, "dynamic"),
            col_count=df_mets.shape[1],
        )
        # On renvoie le composant pour que les autres onglets puissent l'utiliser
        return level_out, wf_comp, wt_comp