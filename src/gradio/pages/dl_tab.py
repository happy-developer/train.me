import gradio as gr

from pathlib import Path
import tensorflow as tf
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "src" / "models" / "v1"

MODEL_REGISTRY = {
    "LSTM": {
        "type": "keras",
        "path": MODEL_DIR / "lstm_wordlevel_v1.keras",
    },
    "Transformer": {
        "type": "keras",
        "path": MODEL_DIR / "transformer_wordlevel_v1.keras",
    },
    "Fine-tuning GPT2": {
        "type": "gpt2",
        "path": MODEL_DIR / "gpt2_trainme_best_finetuned",
    },
    "Distillation GPT2": {
        "type": "gpt2",
        "path": MODEL_DIR / "gpt2_trainme_best_student_distilled",
    },
}

# cache des modèles chargés
LOADED_MODELS = {}

def on_model_change(model_name: str) -> str:
    """Charge le modèle sélectionné et retourne le chemin à afficher."""
    info = MODEL_REGISTRY[model_name]
    path = info["path"]

    if model_name not in LOADED_MODELS:
        if info["type"] == "keras":
            LOADED_MODELS[model_name] = tf.keras.models.load_model(path)
        elif info["type"] == "gpt2":
            # Ici tu pourras aussi gérer tokenizer si besoin
            LOADED_MODELS[model_name] = AutoModelForCausalLM.from_pretrained(path)

    # valeur renvoyée dans le textbox read-only
    return str(path)


def render_dl_tab(app_desc_dl: str) -> None:
    """Onglet Deep Learning."""
    default_model = "Fine-tuning GPT2"

    with gr.Tab(f"Deep Learning - {app_desc_dl}"):

        gr.Markdown(
            "## Generate your personalized exercise\n"
            "Prediction based on your information and the program selected from the list"
        )

        model_selector = gr.Dropdown(
            label="Deep Learning model",
            interactive=True,
            choices=[
                "LSTM",
                "Transformer",
                "Fine-tuning GPT2",
                "Distillation GPT2",
            ],
            value=default_model,
        )

        # Textbox read-only qui affiche le fichier / dossier chargé
        model_path_box = gr.Textbox(
            label="Loaded model path",
            value=str(MODEL_REGISTRY[default_model]["path"]),
            interactive=False,
        )

        prompt_box = gr.Textbox(
            label="Prompt",
            interactive=True,
            lines=4,
            placeholder=(
                "Example: Generate a 45-minute full-body program for a beginner,"
                "3 times a week, with light dumbbells and lower-body work."
            ),
        )

        generated_text = gr.Textbox(
            label="Generated program",
            lines=20,
            max_lines=40,
        )

        # liaison sélection → chargement modèle + maj du chemin affiché
        model_selector.change(
            fn=on_model_change,
            inputs=model_selector,
            outputs=model_path_box,
        )

    return {
        "model_selector": model_selector,
        "model_path_box": model_path_box,
        "prompt_box": prompt_box,
        "generated_text": generated_text,
    }
