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


import textwrap
import torch
import re
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# Définition des chemins principaux
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR_LOCAL = PROJECT_ROOT / "models" / "v1"


# Chemin NAS en Path (UNC)
NAS_ROOT = Path(r"\\SYNONAS-HOME\Projets\Models\trAIn.me")

# Si tu veux forcer le NAS :
MODEL_DIR_NAS = NAS_ROOT


# ---------------------------------------------------------------------
# Registre des modèles DL
# ---------------------------------------------------------------------

MODEL_REGISTRY = {
    # ============================
    # GPT-2 SIMPLE (TF1) — LOCAL
    # ============================
    "xMas - GPT2 Fine-tuning [Run1]": {
        "type": "gpt2_xmas",
        "path": MODEL_DIR_LOCAL / "gpt2-xmas-finetuning_run1",
    },
    "xMas - GPT2 Fine-tuning [Run2]": {
        "type": "gpt2_xmas",
        "path": MODEL_DIR_LOCAL / "gpt2-xmas-finetuning_run2",
    },
    "xMas - GPT2 Fine-tuning [Run3]": {
        "type": "gpt2_xmas",
        "path": MODEL_DIR_LOCAL / "gpt2-xmas-finetuning_run3",
        "report_path": MODEL_DIR_LOCAL / "GPT2_Fine_Tuning_model_report_gpt_2_simple.json",
    },

    # ============================
    # GPT-2 FINE-TUNING — NAS
    # ============================
    "GPT-2 Fine-tuned (HF) v1": {
        "type": "gpt2_fine_tuning",
        "path": MODEL_DIR_NAS / "gpt2_trainme_fine_tuning_gpt2_v1",
    },
    "GPT-2 Fine-tuned (HF) v2": {
        "type": "gpt2_fine_tuning",
        "path": MODEL_DIR_NAS / "gpt2_trainme_fine_tuning_gpt2_v2",
    },
    "GPT-2 Fine-tuned (HF) v3": {
        "type": "gpt2_fine_tuning",
        "path": MODEL_DIR_NAS / "gpt2_trainme_fine_tuning_gpt2_v3",
    },
    "GPT-2 Fine-tuned (HF) v4": {
        "type": "gpt2_fine_tuning",
        "path": MODEL_DIR_NAS / "gpt2_trainme_fine_tuning_gpt2_v4",
    },

    # ============================
    # GPT-2 DISTILLATION — NAS
    # ============================
    "GPT-2 Distilled v1": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v1",
    },
    "GPT-2 Distilled v2": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v2",
    },
    "GPT-2 Distilled v3": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v3",
    },
    "GPT-2 Distilled v5": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v5",
    },
    "GPT-2 Medium Distilled v6": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2-medium_trainme_distillation_gpt2_v6",
    },
    # # Problème
    # "GPT-2 Large Distilled v7": {
    #     "type": "gpt2_distilled",
    #     "path": MODEL_DIR_NAS / "gpt2-large_trainme_distillation_gpt2_v7",
    # },
    "GPT-2 Distilled v8": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v8",
    },
    "GPT-2 Distilled v9": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v9",
    },
    "GPT-2 Distilled v10": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v10",
    },
     "GPT-2 Distilled v14 (8 Epochs)": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v14_epochs8",
    },
     "GPT-2 Distilled v15 (5 Epochs)": {
        "type": "gpt2_distilled",
        "path": MODEL_DIR_NAS / "gpt2_trainme_distillation_gpt2_v15_epochs5",
    },
    # ============================
    # Transformer — NAS
    # ============================
    "Transformer v1": {
        "type": "transformer",
        "path": MODEL_DIR_NAS / "Transformer_v1",
    },
    "Transformer v2": {
        "type": "transformer",
        "path": MODEL_DIR_NAS / "Transformer_v2",
    },
    # ============================
    # LSTM — NAS
    # ============================
    # "LSTM v1": {
    #     "type": "lstm",
    #     "path": MODEL_DIR_NAS / "LSTM_v1" / "lstm_wordlevel_v1.keras",
    # },
    "LSTM v2": {
        "type": "lstm",
        "path": MODEL_DIR_NAS / "LSTM_v2" / "lstm_v2.pt",
    },
    "LSTM v3": {
        "type": "lstm",
        "path": MODEL_DIR_NAS / "LSTM_v3" / "lstm_v3.pt",
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
    model_type = info["type"]
    model_path = info["path"]
    if isinstance(model_path, str):
        model_path = Path(model_path)
    model_path = model_path.resolve()

    # Déjà chargé ? On renvoie direct.
    if model_name in LOADED_MODELS:
        return f"Model loaded from cache: {model_path}"

    # ===========================
    # 1) GPT-2 xMas (TF1 local)
    # ===========================
    if model_type == "gpt2_xmas":
        checkpoint_dir = str(model_path.parent)
        run_name = model_path.name

        tf.reset_default_graph()
        sess = gpt2.start_tf_sess()
        gpt2.load_gpt2(
            sess,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
        )

        LOADED_MODELS[model_name] = {
            "type": model_type,
            "sess": sess,
            "checkpoint_dir": checkpoint_dir,
            "run_name": run_name,
        }
        return f"GPT-2 xMas loaded: {model_path}"

    # ===========================
    # 2) GPT-2 fine-tuning (HF)
    # ===========================
    if model_type == "gpt2_fine_tuning":
        generator = GPT2_FineTuningTextGenerator.get_instance(model_path)
        LOADED_MODELS[model_name] = {
            "type": model_type,
            "generator": generator,
        }
        return f"GPT-2 fine-tuned (HF) loaded: {model_path}"

    # ===========================
    # 3) GPT-2 distillé
    # ===========================
    if model_type == "gpt2_distilled":
        generator = GPT2_DistilledTextGenerator.get_instance(model_path)
        LOADED_MODELS[model_name] = {
            "type": model_type,
            "generator": generator,
        }
        return f"Distilled GPT-2 loaded: {model_path}"


    # ===========================
    # 4) Transformer
    # ===========================
    if model_type == "transformer":
        generator = TransformerTextGenerator.get_instance(model_path)
        LOADED_MODELS[model_name] = {
            "type": model_type,
            "generator": generator,
        }
        return f"Transformer loaded: {model_path}"

    # ===========================
    # 5) LSTM
    # ===========================
    if model_type == "lstm":
        generator = LSTMTextGenerator.get_instance(model_path)
        LOADED_MODELS[model_name] = {
            "type": model_type,
            "generator": generator,
        }
        return f"LSTM loaded: {model_path}"

    raise ValueError(f"Unknown model type '{model_type}' for '{model_name}'")



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
    """
    Génère du texte avec le modèle sélectionné à partir du prompt.

    - gpt2_xmas        → GPT-2 simple (TF1, gpt_2_simple)
    - gpt2_distilled   → GPT-2 distillé (HF, NAS)
    - gpt2_fine_tuning → GPT-2 fine-tuné (HF, NAS)
    - transformer      → Transformer Keras/HF
    - lstm             → LSTM (Keras ou PyTorch)
    """
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a prompt before generating."

    info = MODEL_REGISTRY.get(model_name)
    if info is None:
        return f"❌ Unknown model: {model_name}"

    model_type = info["type"]
    model_path = info["path"]

    # ------------------------------------------------------
    # 1) GPT-2 simple (TF1, gpt_2_simple)
    # ------------------------------------------------------
    if model_type == "gpt2_xmas":
        cache = LOADED_MODELS.get(model_name)

        # Si pas encore chargé (au cas où on n'est pas passé par on_model_change)
        if cache is None:
            status = on_model_change(model_name)
            cache = LOADED_MODELS.get(model_name)
            if cache is None:
                return f"❌ Unable to load GPT-2 model: {status}"

        sess = cache["sess"]
        checkpoint_dir = cache["checkpoint_dir"]
        run_name = cache["run_name"]

        print(f"[GPT-2 TF1] Using checkpoint_dir={checkpoint_dir} run_name={run_name}")

        # On réutilise ta fonction de nettoyage dédiée GPT-2 simple
        text = generate_clean_program(
            sess,
            prompt=prompt,
            max_length=220,
            temperature=0.7,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
        )
        return text

    # ------------------------------------------------------
    # 2) GPT-2 distillation (HF local / NAS)
    # ------------------------------------------------------
    if model_type == "gpt2_distilled":
        cache = LOADED_MODELS.get(model_name)
        generator = cache.get("generator") if cache else None

        if generator is None:
            generator = GPT2_DistilledTextGenerator.get_instance(model_path)
            LOADED_MODELS[model_name] = {"generator": generator}

        return generator.generate_text(prompt)

    # ------------------------------------------------------
    # 3) GPT-2 fine-tuning (HF local / NAS)
    # ------------------------------------------------------
    if model_type == "gpt2_fine_tuning":
        cache = LOADED_MODELS.get(model_name)
        generator = cache.get("generator") if cache else None

        if generator is None:
            generator = GPT2_FineTuningTextGenerator.get_instance(model_path)
            LOADED_MODELS[model_name] = {"generator": generator}

        return generator.generate_text(prompt)

    # ------------------------------------------------------
    # 4) Transformer (Keras / HF)
    # ------------------------------------------------------
    if model_type == "transformer":
        cache = LOADED_MODELS.get(model_name)
        generator = cache.get("generator") if cache else None

        if generator is None:
            generator = TransformerTextGenerator.get_instance(model_path)
            LOADED_MODELS[model_name] = {"generator": generator}

        # Adapter aux signatures de ton TransformerTextGenerator
        return generator.generate_text(
            seed_text=prompt,
            num_words=80,
            temperature=0.9,
        )

    # ------------------------------------------------------
    # 5) LSTM (Keras ou PyTorch)
    # ------------------------------------------------------
    if model_type == "lstm":
        cache = LOADED_MODELS.get(model_name)
        generator = cache.get("generator") if cache else None

        if generator is None:
            generator = LSTMTextGenerator.get_instance(model_path)
            LOADED_MODELS[model_name] = {"generator": generator}

        return generator.generate_text(
            seed_text=prompt,
            num_words=80,
            temperature=0.8,
        )

    # ------------------------------------------------------
    # 6) Type inconnu
    # ------------------------------------------------------
    return f"Unknown model type: {model_type}"



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
