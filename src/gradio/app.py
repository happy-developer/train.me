from pathlib import Path
import gradio as gr
import pandas as pd


# ---------- Paths ----------
from .config import build_paths, UI_EXAMPLES

HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]

p = build_paths(SRC_DIR)

MODEL_DIR   = p["MODEL_DIR"]
MODEL_PATH  = p["MODEL_PATH"]
SCHEMA_PATH = p["SCHEMA_PATH"]
LOGS_DIR    = p["LOGS_DIR"]; LOGS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH     = p["DB_PATH"]
REPORT_PATH = p["REPORT_PATH"]


# ---------- Load model & schema ----------
from .model_loader import load_model_and_schema

model, schema, TARGET_NAME, FEATURES, EXPECTED_ORDER = load_model_and_schema(
    MODEL_PATH, SCHEMA_PATH
)


# ---------- Helpers ----------
from .helpers.log_utils import log_prediction
from .helpers.predict_utils import predict_single
from .helpers.schema_utils import get_bounds
from .helpers.report_utils import read_model_report, report_summary_df, report_metrics_df
from .helpers.sqlite_utils import load_val_subset


# ---------- UI ----------
def build_app():
    app_title = f"TrAIn.me — {schema.get('model_name','model')} ({schema.get('model_version','v?')})"
    app_desc = f"Prédiction de `{TARGET_NAME}` à partir de : {', '.join(EXPECTED_ORDER)}."

    # En-têtes du tableau de validation : cible d'abord, puis features
    DISPLAY_HEADERS = [TARGET_NAME] + [c for c in EXPECTED_ORDER if c != TARGET_NAME]

    with gr.Blocks(title=app_title) as demo:
        gr.Markdown(f"# {app_title}\n{app_desc}")

        # ====== Ligne principale (inputs/pred) ======
        with gr.Row():
            # ---------- COLONNE GAUCHE ----------
            with gr.Column():
                gr.Markdown("### Paramètres d’entrée")
                comps, names = [], []
                for spec in FEATURES:
                    name = spec["name"]
                    vmin, vmax, default, step = get_bounds(spec, schema)
                    comp = gr.Slider(vmin, vmax, value=default, step=step, label=name)
                    comps.append(comp)
                    names.append(name)

                btn = gr.Button("Prédire", variant="primary")

            # ---------- COLONNE DROITE ----------
            with gr.Column():
                gr.Markdown("### Prédiction")
                y_out = gr.Number(label=f"{TARGET_NAME}", interactive=False, precision=2)
                meta_out = gr.Textbox(label="Infos", interactive=False)

        # ====== Exemples (depuis config.py, fallback schéma si vide) ======
        examples_dicts = UI_EXAMPLES or [schema.get("example_payload", {})]
        # On aligne l’ordre des valeurs sur les composants visibles (names)
        examples_rows = [[ex.get(col, "") for col in names] for ex in examples_dicts]

        if any(any(str(v) != "" for v in row) for row in examples_rows):
            gr.Examples(examples=examples_rows, inputs=comps, label="Exemples")

        gr.Markdown("---")

        # ====== DataTable de validation ======
        gr.Markdown(f"### Échantillon validation — colonnes ({', '.join(DISPLAY_HEADERS)})")
        table = gr.Dataframe(
            headers=DISPLAY_HEADERS,
            value=pd.DataFrame(columns=DISPLAY_HEADERS),
            interactive=False,
            wrap=True,
            label="Validation (features + cible si dispo)",
            row_count=(0, "dynamic"),
            col_count=len(DISPLAY_HEADERS),
            datatype=["number"] * len(DISPLAY_HEADERS)
        )
        refresh_btn = gr.Button("Recharger les données 🔄")

        def _load_table():
            df = load_val_subset(DB_PATH, EXPECTED_ORDER, TARGET_NAME, limit=500)
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Si la colonne contient des valeurs non convertibles, on la laisse telle quelle
                    pass
            return df

        # Chargement auto au démarrage + refresh manuel
        demo.load(fn=_load_table, inputs=None, outputs=table)
        refresh_btn.click(fn=_load_table, inputs=None, outputs=table)

        # ====== Prédiction ======
        def _fn(*vals):
            payload = {k: v for k, v in zip(names, vals)}
            return predict_single(
                payload=payload,
                model=model,
                expected_order=EXPECTED_ORDER,
                log_dir=LOGS_DIR,
                model_path=MODEL_PATH,
                schema=schema,
                target_name=TARGET_NAME,
            )

        btn.click(fn=_fn, inputs=comps, outputs=[y_out, meta_out])

        gr.Markdown("---")

        # ====== Rapport modèle (PLEINE LARGEUR, à la fin) ======
        with gr.Row():
            with gr.Column():
                rep = read_model_report(REPORT_PATH)
                df_sum = report_summary_df(rep)
                df_mets = report_metrics_df(rep)

                gr.Markdown("### Rapport modèle")
                summary_tbl = gr.Dataframe(
                    value=df_sum,
                    interactive=False,
                    wrap=True,
                    label="Résumé",
                    row_count=(0, "dynamic"),
                    col_count=df_sum.shape[1]
                )
                metrics_tbl = gr.Dataframe(
                    value=df_mets,
                    interactive=False,
                    wrap=True,
                    label="Métriques par modèle",
                    row_count=(0, "dynamic"),
                    col_count=df_mets.shape[1]
                )

    return demo

