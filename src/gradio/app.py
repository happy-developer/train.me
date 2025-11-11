from __future__ import annotations
import os, json, time, csv, sqlite3
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

# DB: chemin absolu fourni, sinon fallback relatif
DB_DEFAULT_ABS = Path(r"C:\Users\fback\Desktop\Projets\Dev\GitHub\train.me\src\data\processed\life_style_data\life_style_data_val.db")
DB_DEFAULT_REL = SRC_DIR / "data" / "processed" / "life_style_data" / "life_style_data_val.db"
DB_PATH = Path(os.getenv("VAL_DB_PATH", str(DB_DEFAULT_ABS if DB_DEFAULT_ABS.exists() else DB_DEFAULT_REL)))


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
        step = 0.1 if (vmax - vmin) <= 20 else 0.5
    return vmin, vmax, default, step


# ---------- SQLite helpers (DataTable) ----------
def _list_tables(conn: sqlite3.Connection) -> list[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    return [r[0] for r in cur.fetchall()]

def _table_has_all_columns(conn: sqlite3.Connection, table: str, wanted: list[str]) -> bool:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    cols = {row[1] for row in cur.fetchall()}  # row[1] = name
    return set(wanted).issubset(cols)

def _load_val_subset(db_path: Path, wanted_cols: list[str], limit: int = 500) -> pd.DataFrame:
    """
    Charge un sous-ensemble depuis SQLite en affichant toujours la cible en 1ère colonne,
    suivie des features du schéma. Si des colonnes manquent dans la table, elles sont ajoutées vides.
    """
    # Colonnes à afficher : target d'abord, puis features (sans doublon)
    display_cols = [TARGET_NAME] + [c for c in wanted_cols if c != TARGET_NAME]

    if not db_path.exists():
        return pd.DataFrame(columns=display_cols)

    with sqlite3.connect(db_path) as conn:
        # 1) Table qui contient toutes les colonnes demandées (target + features)
        for tbl in _list_tables(conn):
            if _table_has_all_columns(conn, tbl, display_cols):
                cols_str = ", ".join([f'"{c}"' for c in display_cols])
                query = f'SELECT {cols_str} FROM "{tbl}" LIMIT {limit}'
                return pd.read_sql_query(query, conn)

        # 2) Sinon, meilleure table partielle (max d'intersection)
        best_tbl, best_cols = None, []
        for tbl in _list_tables(conn):
            cur = conn.execute(f"PRAGMA table_info('{tbl}')")
            cols = {row[1] for row in cur.fetchall()}
            inter = [c for c in display_cols if c in cols]
            if len(inter) > len(best_cols):
                best_tbl, best_cols = tbl, inter

        if best_tbl:
            cols_str = ", ".join([f'"{c}"' for c in best_cols])
            query = f'SELECT {cols_str} FROM "{best_tbl}" LIMIT {limit}'
            df = pd.read_sql_query(query, conn)

            # Complète les colonnes manquantes (target/feature) et réordonne
            for c in display_cols:
                if c not in df.columns:
                    df[c] = pd.NA
            return df[display_cols]

    # 3) Aucun tableau exploitable
    return pd.DataFrame(columns=display_cols)




# ---------- UI ----------
def build_app():
    app_title = f"TrAIn.me — {schema.get('model_name','model')} ({schema.get('model_version','v?')})"
    app_desc = f"Prédiction de `{TARGET_NAME}` à partir de : {', '.join(EXPECTED_ORDER)}."

    with gr.Blocks(title=app_title) as demo:
        gr.Markdown(f"# {app_title}\n{app_desc}")

        # Inputs
        with gr.Row():
            with gr.Column():
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

        gr.Markdown("---")

        # ======= DataTable (validation set depuis SQLite) =======
        gr.Markdown(f"### Échantillon validation — colonnes du schéma ({', '.join(EXPECTED_ORDER)})")
        table = gr.Dataframe(
            headers=EXPECTED_ORDER,
            value=pd.DataFrame(columns=EXPECTED_ORDER),   # ← évite la ligne 1|2
            interactive=False,
            wrap=True,
            label="Validation (features only)",
            row_count=(0, "dynamic"),
            col_count=len(EXPECTED_ORDER),
            datatype=["number"] * len(EXPECTED_ORDER)
        )
        
        refresh_btn = gr.Button("Recharger les données 🔄")

        def _load_table():
            df = _load_val_subset(DB_PATH, EXPECTED_ORDER, limit=500)
            # assure l'ordre + types numériques si possible
            for col in EXPECTED_ORDER:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="ignore")
            return df

        # Charger au démarrage
        demo.load(fn=_load_table, inputs=None, outputs=table)
        # Bouton refresh
        refresh_btn.click(fn=_load_table, inputs=None, outputs=table)

        # Handler prédiction
        def _fn(*vals):
            payload = {k: v for k, v in zip(names, vals)}
            return _predict(payload)

        btn.click(fn=_fn, inputs=comps, outputs=[y_out, meta_out])

    return demo

