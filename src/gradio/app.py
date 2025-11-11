from __future__ import annotations
import os, json, time, csv
from pathlib import Path

import joblib
import gradio as gr
import pandas as pd


# ---------- Paths ----------
HERE = Path(__file__).resolve()
SRC_DIR = HERE.parents[1]
MODEL_DIR = SRC_DIR / "models" / "v1" / "life_style_data"
MODEL_PATH = Path(os.getenv("MODEL_PATH", MODEL_DIR / "model.joblib"))
SCHEMA_PATH = Path(os.getenv("SCHEMA_PATH", MODEL_DIR / "feature_schema.json"))
LOGS_DIR = SRC_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


# ---------- Load model & schema ----------
model = joblib.load(MODEL_PATH)
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)

TARGET_NAME = schema.get("target", "Calories_Burned")
FEATURES = schema.get("features", [])

# Ordre attendu par le modèle
if hasattr(model, "feature_names_in_"):
    EXPECTED_ORDER = list(model.feature_names_in_)
else:
    EXPECTED_ORDER = [f["name"] for f in FEATURES]


# ---------- Helpers ----------
def _log_prediction(row_in: dict, y_hat: float, latency_ms: int):
    log_file = LOGS_DIR / "predictions.csv"
    write_header = not log_file.exists()
    new_row = {
        "ts": pd.Timestamp.utcnow().isoformat(),
        **row_in,
        TARGET_NAME: y_hat,
        "latency_ms": latency_ms,
        "model_file": MODEL_PATH.name,
        "model_version": schema.get("model_version", "unknown"),
    }
    with log_file.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(new_row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(new_row)


def _predict(payload: dict):
    t0 = time.time()

    # Garde-fous : vérifier les champs
    missing = [c for c in EXPECTED_ORDER if c not in payload]
    if missing:
        raise ValueError(f"Champs manquants dans l'input UI : {missing}")

    # Construire X dans l'ordre attendu + forcer numérique
    X_one = pd.DataFrame([[payload[c] for c in EXPECTED_ORDER]], columns=EXPECTED_ORDER)
    X_one = X_one.apply(pd.to_numeric, errors="raise")

    y_pred = float(model.predict(X_one).squeeze())
    y_pred = round(y_pred, 2)
    latency_ms = int((time.time() - t0) * 1000)

    _log_prediction({k: payload[k] for k in EXPECTED_ORDER}, y_pred, latency_ms)
    meta = f"Latency: {latency_ms} ms | Model: {MODEL_PATH.name} | Version: {schema.get('model_version','?')}"
    return y_pred, meta


def _bounds(spec: dict):
    """Bornes sûres même si le schéma est incomplet."""
    t = spec.get("type", "number")
    if t in ("integer", "int"):
        vmin = int(spec.get("min", 0))
        vmax = int(spec.get("max", 100))
        default = int(schema.get("example_payload", {}).get(spec["name"], (vmin + vmax) // 2))
        step = 1
    else:
        vmin = float(spec.get("min", 0.0))
        vmax = float(spec.get("max", 100.0))
        default = float(schema.get("example_payload", {}).get(spec["name"], (vmin + vmax) / 2))
        # step simple : fin mais lisible
        step = 0.1 if (vmax - vmin) <= 20 else 0.5
    return vmin, vmax, default, step


# ---------- UI ----------
def build_app():
    app_title = f"TrAIn.me — {schema.get('model_name','model')} ({schema.get('model_version','v?')})"
    app_desc = f"Prédiction de `{TARGET_NAME}` à partir de : {', '.join(EXPECTED_ORDER)}."

    with gr.Blocks(title=app_title) as demo:
        gr.Markdown(f"# {app_title}\n{app_desc}")

        # Crée les inputs **dans** le contexte Blocks
        with gr.Row():
            with gr.Column():
                # Inputs depuis le schéma (ici: Age, Weight (kg))
                comps = []
                names = []
                for spec in FEATURES:
                    name = spec["name"]
                    vmin, vmax, default, step = _bounds(spec)
                    comp = gr.Slider(vmin, vmax, value=default, step=step, label=name)
                    comps.append(comp)
                    names.append(name)

                btn = gr.Button("Prédire 🔥", variant="primary")

            with gr.Column():
                y_out = gr.Number(label=f"{TARGET_NAME} (prédiction)", interactive=False, precision=2)
                meta_out = gr.Textbox(label="Infos", interactive=False)

        # Exemple (depuis le schéma)
        ex = schema.get("example_payload", {})
        example_row = [[ex.get(col, "") for col in EXPECTED_ORDER]]
        if any(str(v) != "" for v in example_row[0]):
            gr.Examples(examples=example_row, inputs=comps, label="Exemple (schéma)")

        # Handler
        def _fn(*vals):
            payload = {k: v for k, v in zip(names, vals)}
            return _predict(payload)

        btn.click(fn=_fn, inputs=comps, outputs=[y_out, meta_out])

    return demo

