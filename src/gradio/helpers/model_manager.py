from pathlib import Path
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from transformers import AutoModelForCausalLM, AutoTokenizer

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


def generate_text_with_model(model_name: str, prompt: str) -> str:
    """Génère du texte avec le modèle sélectionné à partir du prompt."""
    prompt = prompt.strip()
    if not prompt:
        return "Please enter a prompt before generating."

    info = MODEL_REGISTRY[model_name]

    # ------------------------------------------------------------------
    # 1) Modèles Keras : LSTM & Transformer
    # ------------------------------------------------------------------
    if info["type"] == "keras":
        if model_name not in LOADED_MODELS:
            on_model_change(model_name)

        if model_name == "LSTM":
            lstm_gen = LSTMTextGenerator.get_instance(LOADED_MODELS[model_name])
            return lstm_gen.generate_text(
                seed_text=prompt,
                num_words=80,
                temperature=0.8,
            )

        if model_name == "Transformer":
            transformer_gen = TransformerTextGenerator.get_instance(
                LOADED_MODELS[model_name]
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

    # ------------------------------------------------------------------
    # 3) Type non géré
    # ------------------------------------------------------------------
    return f"Text generation is not implemented yet for model '{model_name}'."
