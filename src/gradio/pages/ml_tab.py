from pathlib import Path
import gradio as gr
import pandas as pd

from ..helpers.schema_utils import get_bounds
from ..helpers.sqlite_utils import load_val_subset
from ..helpers.predict_utils import predict_single
from ..helpers.report_utils import (
    read_model_report, report_summary_df, report_metrics_df
)


def render_ml_tab(
    app_desc_ml: str,
    features: list[dict],
    expected_order: list[str],
    target_name: str,
    schema: dict,
    ui_examples: list[dict],
    db_path: Path,
    model,
    logs_dir: Path,
    model_path: Path,
    feature_scaler=None,
    target_scaler=None,
    report_path: Path | None = None,
    on_load=None,                      # <- nouveau paramètre
) -> None:
    """Construit l’onglet Machine Learning (UI + callbacks)."""
    display_headers = [target_name] + [c for c in expected_order if c != target_name]

    with gr.Tab(f"Machine Learning - {app_desc_ml}"):
        # ====== Ligne principale (inputs/pred) ======
        with gr.Row():
            # Gauche : sliders
            with gr.Column():
                gr.Markdown("### Paramètres d’entrée")
                comps, names = [], []
                for spec in features:
                    name = spec["name"]
                    vmin, vmax, default, step = get_bounds(spec, schema)
                    comp = gr.Slider(vmin, vmax, value=default, step=step, label=name)
                    comps.append(comp)
                    names.append(name)
                btn = gr.Button("Prédire", variant="primary")

            # Droite : résultats
            with gr.Column():
                gr.Markdown("### Prédiction")
                y_out = gr.Number(label=target_name, interactive=False, precision=2)
                meta_out = gr.Textbox(label="Infos", interactive=False)

        # ====== Exemples ======
        examples_dicts = ui_examples or [schema.get("example_payload", {})]
        rows = [[ex.get(col, "") for col in names] for ex in examples_dicts]
        if any(any(str(v) != "" for v in r) for r in rows):
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
            df = load_val_subset(db_path, expected_order, target_name, limit=500)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
            return df

        # Enregistrement du chargement auto fourni par le parent
        if on_load is not None:
            on_load(fn=_load_table, inputs=None, outputs=table)

        refresh_btn.click(_load_table, None, table)

        # ====== Prédiction ======
        def _fn(*vals):
            payload = {k: v for k, v in zip(names, vals)}
            return predict_single(
                payload=payload,
                model=model,
                expected_order=expected_order,
                log_dir=logs_dir,
                model_path=model_path,
                schema=schema,
                target_name=target_name,
                feature_scaler=feature_scaler,
                target_scaler=target_scaler,
            )

        btn.click(_fn, comps, [y_out, meta_out])

        gr.Markdown("---")

        # ====== Rapport modèle ======
        if report_path:
            with gr.Row():
                with gr.Column():
                    rep = read_model_report(report_path)
                    df_sum = report_summary_df(rep)
                    df_mets = report_metrics_df(rep)

                    gr.Markdown("### Rapport modèle")
                    gr.Dataframe(value=df_sum, interactive=False, label="Résumé",
                                 row_count=(0, "dynamic"), col_count=df_sum.shape[1])
                    gr.Dataframe(value=df_mets, interactive=False, label="Métriques par modèle",
                                 row_count=(0, "dynamic"), col_count=df_mets.shape[1])
