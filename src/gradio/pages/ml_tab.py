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
    gender_encoder,
    report_path: Path,
    on_load=None,
) -> None:
    display_headers = [target_name] + [c for c in ui_feature_names if c != target_name]

    with gr.Tab(f"Machine Learning - {app_desc_ml}"):
        # ====== Ligne principale (inputs/pred) ======
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Paramètres d’entrée")
                comps, names = [], []

                for spec in feature_specs:
                    name = spec["name"]
                    if name == "Gender":
                        # UI = Radio pour Male / Female
                        choices = spec.get("enum", ["Male", "Female"])
                        comp = gr.Radio(choices=choices, value=choices[0], label="Gender")
                    else:
                        vmin, vmax, default, step = get_bounds(spec, schema)
                        comp = gr.Slider(vmin, vmax, value=default, step=step, label=name)

                    comps.append(comp)
                    names.append(name)

                btn = gr.Button("Prédire", variant="primary")

            with gr.Column():
                gr.Markdown("### Prédiction")
                y_out = gr.Number(label=target_name, interactive=False, precision=2)
                meta_out = gr.Textbox(label="Infos", interactive=False)

        # ====== Exemples ======
        examples_dicts = ui_examples or [schema.get("example_payload", {})]
        rows = [[ex.get(col, "") for col in names] for ex in examples_dicts]
        if any(any(str(v) != "" for v in row) for row in rows):
            gr.Examples(examples=rows, inputs=comps, label="Exemples")

        gr.Markdown("---")

        # ====== DataTable de validation ======
        gr.Markdown(f"### Échantillon validation — colonnes ({', '.join(display_headers)})")
        table = gr.Dataframe(
            headers=display_headers,
            value=pd.DataFrame(columns=display_headers),
            interactive=False,
            wrap=True,
            label="Validation (features + cible si dispo)",
            row_count=(0, "dynamic"),
            col_count=len(display_headers),
            datatype=["number"] * len(display_headers),
        )
        refresh_btn = gr.Button("Recharger les données 🔄")

        def _load_table():
            # On reste sur les colonnes “métier” (Age, Weight (kg), Gender, cible)
            df = load_val_subset(db_path, ui_feature_names, target_name, limit=500)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            return df

        if on_load is not None:
            on_load(fn=_load_table, inputs=None, outputs=table)
        refresh_btn.click(_load_table, None, table)

        # ====== Prédiction ======
        def _fn(*vals):
            payload = {k: v for k, v in zip(names, vals)}

            # Gradio peut renvoyer les valeurs numériques en str → on force
            if "Age" in payload:
                payload["Age"] = float(payload["Age"])
            if "Weight (kg)" in payload:
                payload["Weight (kg)"] = float(payload["Weight (kg)"])
            if "Experience_Level" in payload:
                payload["Experience_Level"] = float(payload["Experience_Level"])

            return predict_single(
                payload=payload,
                internal_expected=internal_expected,
                model=model,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
                log_dir=logs_dir,
                model_path=model_path,
                schema=schema,
                target_name=target_name,
                gender_encoder=gender_encoder,
            )

        btn.click(_fn, comps, [y_out, meta_out])

        gr.Markdown("---")

        # ====== Rapport modèle ======
        rep = read_model_report(report_path)
        df_sum = report_summary_df(rep)
        df_mets = report_metrics_df(rep)

        gr.Markdown("### Rapport modèle")
        gr.Dataframe(
            value=df_sum,
            interactive=False,
            wrap=True,
            label="Résumé",
            row_count=(0, "dynamic"),
            col_count=df_sum.shape[1],
        )
        gr.Dataframe(
            value=df_mets,
            interactive=False,
            wrap=True,
            label="Métriques par modèle",
            row_count=(0, "dynamic"),
            col_count=df_mets.shape[1],
        )
