from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..helpers.custom_layers import MultiHeadSelfAttention, PositionalEmbedding, TransformerBlock

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
        "type": "keras",
        "path": MODEL_DIR / "lstm_wordlevel_v1.keras",
    },
    "Transformer": {
        "type": "keras",
        "path": MODEL_DIR / "transformer_wordlevel_v1.keras",
    },
    "GPT2 Fine-tuning": {
        "type": "gpt2",
        "path": MODEL_DIR / "gpt2_trainme_best_finetuned",
    },
    "GPT2 Distillation": {
        "type": "gpt2",
        "path": MODEL_DIR / "gpt2_trainme_best_student_distilled",
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

    if info["type"] == "keras":
        # Par défaut, on reste en safe_mode pour les modèles simples (LSTM)
        custom_objects = {}
        safe_mode = True

        # Transformer : couches custom + Lambda → safe_mode désactivé
        if model_name == "Transformer":
            custom_objects["PositionalEmbedding"] = PositionalEmbedding
            custom_objects["TransformerBlock"] = TransformerBlock
            custom_objects["MultiHeadSelfAttention"] = MultiHeadSelfAttention
            safe_mode = False

        model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects or None,  # None pour LSTM
            safe_mode=safe_mode,
            compile=False,  # on ne réentraîne pas dans Gradio, juste inférence
        )
        LOADED_MODELS[model_name] = model

    elif info["type"] == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_path)
        LOADED_MODELS[model_name] = model

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        LOADED_TOKENIZERS[model_name] = tokenizer

    else:
        raise ValueError(f"Unsupported model type: {info['type']}")

    return f"Model loaded: {model_path}"
