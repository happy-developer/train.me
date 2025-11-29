import gradio as gr

# Import du helper que tu viens de créer
from ..helpers.select_nlp_model import on_model_change


def render_dl_tab(app_desc_dl: str) -> None:
    """Onglet Deep Learning (sélection du modèle + prompt + sortie texte)."""
    with gr.Tab(f"Deep Learning - {app_desc_dl}"):

        # Titre + sous-texte
        gr.Markdown(
            "## Generate your personalized exercise\n"
            "Prediction based on your information and the program selected from the list"
        )

        # Sélecteur du modèle DL
        model_selector = gr.Dropdown(
            label="Deep Learning model",
            choices=[
                "LSTM",
                "Transformer",
                "GPT2 Fine-tuning",
                "GPT2 Distillation",
            ],
            value="GPT2 Fine-tuning",
        )

        # Zone d'info sur le modèle chargé (affiche le chemin, cache, etc.)
        model_status = gr.Markdown(
            "No model loaded yet. Select one from the list above.",
        )

        # Champ Prompt (éditable)
        prompt_box = gr.Textbox(
            label="Prompt",
            interactive=True,
            lines=4,
            placeholder=(
                "Exemple : Generate a 45-minute full body workout for a beginner, "
                "3 times per week, with light dumbbells and focus on lower body."
            ),
        )

        # Zone d’affichage du texte généré (programme) – >2000 caractères OK
        generated_text = gr.Textbox(
            label="Generated program",
            lines=20,
            max_lines=40,
        )

        # --- Callbacks ---

        # Quand on change de modèle, on charge via on_model_change()
        # et on affiche le message de statut dans model_status
        model_selector.change(
            fn=on_model_change,
            inputs=model_selector,
            outputs=model_status,
        )

    # Retour des composants si tu veux les réutiliser plus tard
    return {
        "model_selector": model_selector,
        "model_status": model_status,
        "prompt_box": prompt_box,
        "generated_text": generated_text,
    }
