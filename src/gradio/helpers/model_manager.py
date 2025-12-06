from pathlib import Path
import pandas as pd
import json
import re
import numpy as np
import tensorflow as tf
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..generators.gpt2_distillation_text_generator import GPT2_DistilledTextGenerator
from ..generators.gpt2_fine_tuning_text_generator import GPT2_FineTuningTextGenerator
from ..generators.transformer_text_generator import TransformerTextGenerator
from ..generators.lstm_text_generator import LSTMTextGenerator  # en haut du fichier si pas déjà fait

from .custom_layers import MultiHeadSelfAttention, PositionalEmbedding, TransformerBlock

# ---------------------------------------------------------------------
# Définition des chemins principaux
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "v1"

# ---------------------------------------------------------------------
# Registre des modèles DL
# ---------------------------------------------------------------------

MODEL_REGISTRY = {
    "LSTM": {
        "type": "pt",
        "path": MODEL_DIR / "lstm_v3.pt",
        "report_path": MODEL_DIR / "LSTM_model_report.json",
    },
    "Transformer": {
        "type": "pt",
        "path": MODEL_DIR / "transformer_v2.pt",
        "report_path": MODEL_DIR / "Transformer_model_report.json",
    },
    "GPT2 Fine-tuning": {
        "type": "gpt2",
        "path": MODEL_DIR / "gpt2_trainme_fine_tuning_gpt2_v3",
        "report_path": MODEL_DIR / "GPT2_Fine_Tuning_model_report.json",
    },
    "GPT2 Distillation": {
        "type": "gpt2",
        # "path": MODEL_DIR / "gpt2_trainme_distillation_gpt2_v5",
        "path": MODEL_DIR / "gpt2-medium_trainme_distillation_gpt2_v6",
        "report_path": MODEL_DIR / "GPT2_Distillation_model_report.json",
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

    # Déjà chargé ? On renvoie direct.
    if model_name in LOADED_MODELS:
        return f"Model loaded from cache: {model_path}"

    model_path_str = str(model_path)

    if info["type"] == "pt":
        # ------------------------------------------------------------------
        # 1) Modèles PyTorch (.pt) : LSTM / Transformer v2
        # ------------------------------------------------------------------
        if model_path_str.endswith(".pt"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = torch.load(model_path, map_location=device)

            # On passe en mode eval si possible
            if hasattr(model, "eval"):
                model.eval()

            LOADED_MODELS[model_name] = model

    elif info["type"] == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        LOADED_MODELS[model_name] = model

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        LOADED_TOKENIZERS[model_name] = tokenizer

    else:
        raise ValueError(f"Unsupported model type: {info['type']}")

    return f"Model loaded: {model_path}"


def generate_text_with_model(model_name: str, prompt: str) -> str:
    """Génère du texte avec le modèle sélectionné à partir du prompt."""
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a prompt before generating."

    info = MODEL_REGISTRY[model_name]

    # ------------------------------------------------------------------
    # 1) Modèles Keras : LSTM & Transformer
    # ------------------------------------------------------------------
    if info["type"] == "pt":
        if model_name not in LOADED_MODELS:
            on_model_change(model_name)

        # ------------------------------------------------------------------
        # 1) LSTM  (PyTorch ou Keras, peu importe pour le wrapper)
        # ------------------------------------------------------------------
        if model_name == "LSTM":
            lstm_gen = LSTMTextGenerator.get_instance(LOADED_MODELS[model_name])
            return lstm_gen.generate_text(
                seed_text=prompt,
                num_words=80,
                temperature=0.8,
            )

        # ------------------------------------------------------------------
        # 2) Transformer (idem, backend abstrait par le wrapper)
        # ------------------------------------------------------------------
        if model_name == "Transformer":
            transformer_gen = TransformerTextGenerator.get_instance(
                LOADED_MODELS[model_name]  # state_dict OU nn.Module
            )
            return transformer_gen.generate_text(
                seed_text=prompt,
                num_words=80,
                temperature=0.9,
            )

        return f"Text generation is not implemented yet for Keras model '{model_name}'."

    # ------------------------------------------------------------------
    # 2) GPT-2 Fine-tuning uniquement
    # ------------------------------------------------------------------
    if info["type"] == "gpt2":
        if model_name not in LOADED_MODELS:
            on_model_change(model_name)

        if model_name == "GPT2 Fine-tuning":
            fine_tuned_gpt2_gen = GPT2_FineTuningTextGenerator(
                model=LOADED_MODELS[model_name],
                tokenizer=LOADED_TOKENIZERS[model_name],
                max_new_tokens=256,
            )

            return fine_tuned_gpt2_gen.generate_text(
                prompt=prompt,
                temperature=0.9,   # réglage recommandé dans ton notebook
                top_p=0.95,
                strip_prompt=True,
            )
        
        if model_name == "GPT2 Distillation":
            distilled_gpt2_gen = GPT2_DistilledTextGenerator(
                model=LOADED_MODELS[model_name],
                tokenizer=LOADED_TOKENIZERS[model_name],
                max_new_tokens=256,
            )
            # return distilled_gpt2_gen.generate_text(
            #     prompt=prompt,
            #     temperature=0.8,   # un poil plus "sage" pour le student
            #     top_p=0.9,
            #     strip_prompt=True,
            # )
            return distilled_gpt2_gen.generer_exercice_interactif(               
                workout_type="cardio",
                debut="To increase your endurance, try to",
                num_samples=2,
            )

    # ------------------------------------------------------------------
    # 3) Type non géré
    # ------------------------------------------------------------------
    return f"Text generation is not implemented yet for model '{model_name}'."


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
    except Exception:
        return _empty_dl_dfs()

    # Summary
    summary_keys = [
        "created_at",
        "task",
        "target",
        "n_train_samples",
        "n_val_samples",
        "vocab_size",
        "sequence_length",
        "device",
    ]
    df_summary = pd.DataFrame(
        [(k, data.get(k, "")) for k in summary_keys],
        columns=["Key", "Value"]
    )

    # Model config
    model_cfg = data.get("model", {})
    df_model = pd.DataFrame(
        [(k, v) for k, v in model_cfg.items()],
        columns=["Key", "Value"]
    )

    # Training
    training_cfg = data.get("training", {})
    df_training = pd.DataFrame(
        [(k, v) for k, v in training_cfg.items()],
        columns=["Key", "Value"]
    )

    # Metrics
    metrics_cfg = data.get("metrics", {})
    df_metrics = pd.DataFrame(
        [(k, v) for k, v in metrics_cfg.items()],
        columns=["Metric", "Value"]
    )

    return df_summary, df_model, df_training, df_metrics


def _empty_dl_dfs():
    """Retourne 4 DataFrames vides pour éviter les erreurs."""
    empty = pd.DataFrame({"Key": [], "Value": []})
    empty_metrics = pd.DataFrame({"Metric": [], "Value": []})
    return empty, empty, empty, empty_metrics
