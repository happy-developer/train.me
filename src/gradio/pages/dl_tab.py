import gradio as gr
import pandas as pd

# Import du helper
from ..helpers.model_manager import (
    on_model_change,
    generate_text_with_model,
    get_dl_model_report_components,
)

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

        gr.Markdown("---")

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

        # Bouton "Générer" centré sous le prompt
        with gr.Row():
            gr.Column(scale=1)  # espace à gauche
            with gr.Column(scale=1):
                generate_btn = gr.Button("Générer")
            gr.Column(scale=1)  # espace à droite

        # Zone d’affichage du texte généré (programme) – >2000 caractères OK
        generated_text = gr.Textbox(
            label="Generated program",
            lines=20,
            max_lines=40,
        )

        # Wiring : clic sur "Générer" → appelle le modèle sélectionné avec le prompt
        generate_btn.click(
            fn=generate_text_with_model,
            inputs=[model_selector, prompt_box],
            outputs=generated_text,
        )

        # Tableaux du rapport DL
        gr.Markdown("### Deep Learning model evaluation report")

        dl_sum = gr.Dataframe(
            value=pd.DataFrame({"Key": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Summary",
        )

        dl_model = gr.Dataframe(
            value=pd.DataFrame({"Key": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Model",
        )

        dl_training = gr.Dataframe(
            value=pd.DataFrame({"Key": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Training",
        )

        dl_metrics = gr.Dataframe(
            value=pd.DataFrame({"Metric": [], "Value": []}),
            interactive=False,
            wrap=True,
            label="Metrics",
        )
       
        # --- Callbacks ---
        def _on_dl_model_select(name: str):
            status = on_model_change(name)
            df_sum, df_model, df_training, df_metrics = get_dl_model_report_components(name)
            return status, df_sum, df_model, df_training, df_metrics
        
        # Quand on change de modèle, on charge via on_model_change()
        # et on affiche le message de statut dans model_status
        model_selector.change(
            fn=_on_dl_model_select,
            inputs=model_selector,
            outputs=[model_status, dl_sum, dl_model, dl_training, dl_metrics],
        )

    # Retour des composants si tu veux les réutiliser plus tard
    return {
        "model_selector": model_selector,
        "model_status": model_status,
        "prompt_box": prompt_box,
        "generated_text": generated_text,
    }
