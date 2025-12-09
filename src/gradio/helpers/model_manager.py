from pathlib import Path
import pandas as pd
import json
import gpt_2_simple as gpt2
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from ..generators.gpt2_distillation_text_generator import GPT2_DistilledTextGenerator
from ..generators.gpt2_fine_tuning_text_generator import GPT2_FineTuningTextGenerator
from ..generators.transformer_text_generator import TransformerTextGenerator
from ..generators.lstm_text_generator import LSTMTextGenerator

# ---------------------------------------------------------------------
# Définition des chemins principaux
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "v1"

# ---------------------------------------------------------------------
# Registre des modèles DL
# ---------------------------------------------------------------------

MODEL_REGISTRY = {
    "xMas - GPT2 Fine-tuning": {
        "type": "gpt2",
        # "path": MODEL_DIR / "gpt2_trainme_distillation_gpt2_v5",
        "path": MODEL_DIR / "gpt2-xmas-finetuning_run3",
        "report_path": MODEL_DIR / "GPT2_Fine_Tuning_model_report_gpt_2_simple.json",
    },
}

# Cache interne pour éviter de recharger plusieurs fois
LOADED_MODELS = {}
LOADED_TOKENIZERS = {}


# ---------------------------------------------------------------------
# Fonction appelée par ton événement Gradio lors du changement de modèle
# ---------------------------------------------------------------------
def on_model_change(model_name: str) -> str:
    """
    Charge le modèle NLP correspondant au nom sélectionné.
    Retourne simplement le chemin du modèle chargé (affiché dans l'UI).
    """

    info = MODEL_REGISTRY[model_name]
    model_path = info["path"]
    if isinstance(model_path, str):
        model_path = Path(model_path)

    checkpoint_dir = str(model_path.parent)
    run_name = model_path.name

    # Déjà chargé ? On renvoie direct.
    if model_name in LOADED_MODELS:
        return f"Model loaded from cache: {model_path}"

    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(
        sess,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
    )

    # On pourrait stocker la session si besoin plus tard
    LOADED_MODELS[model_name] = {
        "sess": sess,
        "checkpoint_dir": checkpoint_dir,
        "run_name": run_name,
    }

    return f"Model loaded: {model_path}"


def clean_special_tokens(text: str) -> str:
    return (
        text.replace("<|startoftext|>", "")
        .replace("<|endoftext|>", "")
        .strip()
    )


def cut_to_last_sentence(text: str, min_chars: int = 60) -> str:
    """
    Coupe le texte au dernier '.', '!' ou '?' pour éviter
    les phrases tronquées. min_chars évite de couper trop tôt.
    """
    last_dot = text.rfind(".")
    last_exc = text.rfind("!")
    last_q = text.rfind("?")
    end_idx = max(last_dot, last_exc, last_q)

    if end_idx != -1 and end_idx >= min_chars:
        return text[: end_idx + 1].strip()
    return text.strip()


def generate_clean_program(
    sess,
    prompt: str,
    max_length: int = 200,
    temperature: float = 0.8,
    checkpoint_dir: str | None = None,
    run_name: str | None = None,
) -> str:
    """
    Génère un texte brut avec gpt_2_simple en forçant checkpoint_dir / run_name,
    puis applique le nettoyage spécifique TrAIn.me.
    """
    gen_kwargs = dict(
        sess=sess,
        prefix=prompt,
        length=max_length,
        temperature=temperature,
        top_k=40,
        nsamples=1,
        batch_size=1,
        return_as_list=True,
        truncate="<|endoftext|>",  # s'arrête si le token apparaît
    )

    # IMPORTANT : on force les chemins si fournis
    if checkpoint_dir is not None:
        gen_kwargs["checkpoint_dir"] = checkpoint_dir
    if run_name is not None:
        gen_kwargs["run_name"] = run_name

    raw_list = gpt2.generate(**gen_kwargs)
    raw = raw_list[0] if isinstance(raw_list, list) else raw_list

    txt = clean_special_tokens(raw)
    txt = cut_to_last_sentence(txt)
    return txt


def generate_text_with_model(model_name: str, prompt: str) -> str:
    """Génère du texte avec le modèle sélectionné à partir du prompt."""
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a prompt before generating."

    info = MODEL_REGISTRY[model_name]
    model_path = info["path"]  # Path vers le dossier qui contient encoder.json / model-xxx
    if isinstance(model_path, str):
        model_path = Path(model_path)

    checkpoint_dir = str(model_path.parent)   # ex: .../src/models/v1
    run_name = model_path.name                # ex: "gpt2-xmas-finetuning_run3"

    print(f"[GPT-2] Using checkpoint_dir={checkpoint_dir} run_name={run_name}")

    # Reset du graphe TF + session
    tf.reset_default_graph()
    sess = gpt2.start_tf_sess()

    # On indique explicitement où est le modèle
    gpt2.load_gpt2(
        sess,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
    )

    text = generate_clean_program(
        sess,
        prompt=prompt,
        max_length=220,
        temperature=0.7,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
    )
    return text


def get_dl_model_report_components(model_name: str):
    """
    Retourne 4 DataFrames Gradio-ready :
    - Summary
    - Model
    - Training
    - Metrics

    Si pas de rapport → retourne les DF vides.
    """
    info = MODEL_REGISTRY.get(model_name)
    if not info:
        return _empty_dl_dfs()

    report_path = info.get("report_path")
    if not report_path or not report_path.exists():
        return _empty_dl_dfs()

    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[DL REPORT] Error while reading {report_path}: {e}")
        return _empty_dl_dfs()

    # ===== Summary =====
    dataset = data.get("dataset", {})

    summary_rows = [
        ("created_at",        data.get("created_at", "")),
        ("task",              data.get("task", "")),
        ("target",            data.get("target", "")),
        ("framework",         data.get("framework", "")),
        ("dataset.file",      dataset.get("file", "")),
        ("dataset.size_bytes", dataset.get("size_bytes", "")),
        ("dataset.tokens",    dataset.get("tokens", "")),
    ]

    df_summary = pd.DataFrame(summary_rows, columns=["Key", "Value"])

    # ===== Model config =====
    model_cfg = data.get("model", {}) or {}
    df_model = pd.DataFrame(
        [(k, v) for k, v in model_cfg.items()],
        columns=["Key", "Value"],
    )

    # ===== Training =====
    training_cfg = data.get("training", {}) or {}
    df_training = pd.DataFrame(
        [(k, v) for k, v in training_cfg.items()],
        columns=["Key", "Value"],
    )

    # ===== Metrics =====
    metrics_cfg = data.get("metrics", {}) or {}
    df_metrics = pd.DataFrame(
        [(k, v) for k, v in metrics_cfg.items()],
        columns=["Metric", "Value"],
    )

    return df_summary, df_model, df_training, df_metrics



def _empty_dl_dfs():
    """Retourne 4 DataFrames vides pour éviter les erreurs."""
    empty = pd.DataFrame({"Key": [], "Value": []})
    empty_metrics = pd.DataFrame({"Metric": [], "Value": []})
    return empty, empty, empty, empty_metrics
