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
FEATURE_SCALER_PATH = p.get("FEATURE_SCALER_PATH")
TARGET_SCALER_PATH  = p.get("TARGET_SCALER_PATH")
GENDER_ENCODER_PATH = p["GENDER_ENCODER_PATH"]
SCHEMA_PATH = p["SCHEMA_PATH"]
LOGS_DIR    = p["LOGS_DIR"]; LOGS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH     = p["DB_PATH"]
REPORT_PATH = p["REPORT_PATH"]


# ---------- Load model & schema ----------
from .model_loader import load_model_and_schema, load_optional_joblib

model, schema, TARGET_NAME, FEATURES, INTERNAL_EXPECTED = load_model_and_schema(MODEL_PATH, SCHEMA_PATH)
fx_scaler = load_optional_joblib(FEATURE_SCALER_PATH)
y_scaler  = load_optional_joblib(TARGET_SCALER_PATH)
gender_encoder   = load_optional_joblib(GENDER_ENCODER_PATH)
UI_FEATURE_NAMES = [f["name"] for f in FEATURES]

# ---------- Helpers ----------
from .helpers.log_utils import log_prediction
from .helpers.predict_utils import predict_single
from .helpers.schema_utils import get_bounds
from .helpers.report_utils import read_model_report, report_summary_df, report_metrics_df
from .helpers.sqlite_utils import load_val_subset


# ---------- UI ----------
def build_app():
    app_title = f"TrAIn.me — {schema.get('model_name','model')} ({schema.get('model_version','v?')})"
    app_desc_ml = f"Prédiction de `{TARGET_NAME}`"
    app_desc_dl = "Génération d'un programme sportif sur demande."

    from .pages.ml_tab import render_ml_tab
    from .pages.dl_tab import render_dl_tab
    from .config import UI_EXAMPLES

    with gr.Blocks(title=app_title) as demo:
        gr.Markdown(f"# {app_title}\n{app_desc_ml} / {app_desc_dl}")
        with gr.Tabs():
            render_ml_tab(
                app_desc_ml=app_desc_ml,
                feature_specs=FEATURES,
                ui_feature_names=UI_FEATURE_NAMES,
                internal_expected=INTERNAL_EXPECTED,
                target_name=TARGET_NAME,
                schema=schema,
                ui_examples=UI_EXAMPLES,
                db_path=DB_PATH,
                model=model,
                logs_dir=LOGS_DIR,
                model_path=MODEL_PATH,
                feature_scaler=fx_scaler,
                target_scaler=y_scaler,
                gender_encoder=gender_encoder,
                report_path=REPORT_PATH,
                on_load=demo.load,
            )
            render_dl_tab(app_desc_dl=app_desc_dl)

    return demo



