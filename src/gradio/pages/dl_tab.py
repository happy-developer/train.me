import gradio as gr
import pandas as pd

# Import du helper
from ..helpers.model_manager import (
    on_model_change,
    generate_text_with_model,
    get_dl_model_report_components,
)


def render_dl_tab(
    app_desc_dl: str,
    level_out: gr.Textbox,
    wf_comp: gr.Component,
    wt_comp: gr.Component,
    selected_program_df: gr.State,
    goal_state: gr.State,
) -> dict:
    """Onglet Deep Learning (sélection du modèle + prompt + sortie texte)."""

    with gr.Tab(f"Deep Learning - {app_desc_dl}") as tab_dl:

        # ===== Rappel du profil utilisateur (venant du tab ML) =====
        gr.Markdown("### Your profile (from Machine Learning tab)")

        with gr.Row():
            level_display = gr.Textbox(
                label="Physical level",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            wf_display = gr.Textbox(
                label="Workout frequency (days/week)",
                interactive=False,
                lines=1,
                max_lines=1,
            )
            wt_display = gr.Textbox(
                label="Workout type",
                interactive=False,
                lines=1,
                max_lines=1,
            )

        gr.Markdown("### Selected program (from List of programs)")

        goal_display = gr.Textbox(
            label="Training goal",
            interactive=False,
            lines=1,
            max_lines=1,
        )

        program_display = gr.Dataframe(
            value=pd.DataFrame(),
            interactive=False,
            wrap=True,
            label="Selected program",
            row_count=(0, "dynamic"),
            col_count=(0, "dynamic"),
        )

        gr.Markdown(
            "These values are synchronized with your Machine Learning profile "
            "and are used to build the Deep Learning prompt automatically."
        )

        gr.Markdown(
            "## Generate your personalized exercise\n"
            "Prediction based on your information and the program selected from the list"
        )

        # Champ Prompt (auto-généré)
        prompt_box = gr.Textbox(
            label="Prompt sent to the Deep Learning model",
            interactive=True,
            lines=4,
        )

        # Sélecteur du modèle DL
        model_selector = gr.Dropdown(
            label="Sélection du modèle Deep Learning",
            choices=[
                # LSTM (NAS)
                # "LSTM v1",
                "LSTM v2",
                "LSTM v3",

                # Transformer (NAS)
                # "Transformer v1",
                "Transformer v2",                     

                # GPT-2 HF fine-tuning (NAS)
                "GPT-2 Fine-tuned (HF) v1",
                "GPT-2 Fine-tuned (HF) v2",
                "GPT-2 Fine-tuned (HF) v3",
                "GPT-2 Fine-tuned (HF) v4",

                # GPT-2 distillation (NAS)
                "GPT-2 Distilled v1",
                "GPT-2 Distilled v2",
                "GPT-2 Distilled v3",
                "GPT-2 Distilled v5",
                "GPT-2 Medium Distilled v6",
                # "GPT-2 Large Distilled v7",
                "GPT-2 Distilled v8",
                "GPT-2 Distilled v9",
                "GPT-2 Distilled v10",
                # "GPT-2 Distilled v11",
                "GPT-2 Distilled v14 (8 Epochs)",
                "GPT-2 Distilled v15 (5 Epochs)",

                # GPT-2 simple (TF1 – local)
                "xMas - GPT2 Fine-tuning [Run1]",
                "xMas - GPT2 Fine-tuning [Run2]",
                "xMas - GPT2 Fine-tuning [Run3]",                
            ],
            value="GPT-2 Distilled v9",  # valeur par défaut (à adapter)
        )


        # Zone d'info sur le modèle chargé
        model_status = gr.Markdown(
            "No model loaded yet. Select one from the list above.",
        )

        gr.Markdown("---")

        # Bouton "Générer" centré
        with gr.Row():
            gr.Column(scale=1)
            with gr.Column(scale=1):
                generate_btn = gr.Button("Générer")
            gr.Column(scale=1)

        # Zone d’affichage du texte généré (programme)
        generated_text = gr.Textbox(
            label="Generated program",
            lines=20,
            max_lines=40,
        )

        # Wiring : clic sur "Générer"
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

        # --- Helpers internes ---

        def _build_prompt(
            level_text: str,
            wf_text: str,
            wt_text: str,
            program_row,
            goal_text: str,
        ) -> str:
            """
            Construit le prompt à partir du profil + programme sélectionné.
            """
            level = level_text or "Unknown"
            wf = wf_text or "N/A"
            wt = wt_text or "Olympic Weightlifting"
            goal = goal_text or "Olympic Weightlifting"

            if not program_row:
                return (
                    "Generate a workout program based on the user's physical level "
                    "and training goal. No specific exercise has been selected yet."
                )

            ex_name = program_row.get("exercise_name") or "the selected exercise"
            target = program_row.get("target_muscles") or "the target muscles"
            equip = program_row.get("equipment") or "bodyweight only"

            prompt = (
                f"Generate a workout program at [{level}] level. "
                f"I am currently training for [{wt}] and my training frequency "
                f"is [{wf}] days per week. "
                f"My main goal is [{goal}]. "
                f"The exercise to generate is titled [{ex_name}], "
                f"it targets the [{target}] and uses the following equipment: "
                f"[{equip}]."
            )
            return prompt

        # --- Callbacks ---

        def _on_dl_model_select(name: str):
            status = on_model_change(name)
            df_sum, df_model, df_training, df_metrics = get_dl_model_report_components(
                name
            )
            return status, df_sum, df_model, df_training, df_metrics

        # Quand on change de modèle manuellement
        model_selector.change(
            fn=_on_dl_model_select,
            inputs=model_selector,
            outputs=[model_status, dl_sum, dl_model, dl_training, dl_metrics],
        )
        
        # Synchronisation du profil + programme + prompt + rapport à l'ouverture du tab DL
        def _sync_profile(level_val, wf_val, wt_val, program_row, goal_val, model_name):
            level_text = level_val or ""
            wf_text = "" if wf_val in (None, "") else str(wf_val)
            wt_text = wt_val or ""
            goal_text = goal_val or "Olympic Weightlifting"

            if not program_row:
                program_df = pd.DataFrame()
            else:
                program_df = pd.DataFrame([program_row])

            prompt = _build_prompt(level_text, wf_text, wt_text, program_row, goal_text)

            # Chargement du modèle + rapport dès ouverture du tab
            status = on_model_change(model_name)
            df_sum, df_model, df_training, df_metrics = get_dl_model_report_components(
                model_name
            )

            return (
                level_text,
                wf_text,
                wt_text,
                goal_text,
                program_df,
                prompt,
                status,
                df_sum,
                df_model,
                df_training,
                df_metrics,
            )

        tab_dl.select(
            _sync_profile,
            inputs=[
                level_out,
                wf_comp,
                wt_comp,
                selected_program_df,
                goal_state,
                model_selector,
            ],
            outputs=[
                level_display,
                wf_display,
                wt_display,
                goal_display,
                program_display,
                prompt_box,
                model_status,
                dl_sum,
                dl_model,
                dl_training,
                dl_metrics,
            ],
        )

    # Retour des composants si besoin
    return {
        "model_selector": model_selector,
        "model_status": model_status,
        "prompt_box": prompt_box,
        "generated_text": generated_text,
        "level_display": level_display,
        "wf_display": wf_display,
        "wt_display": wt_display,
        "goal_display": goal_display,
        "program_display": program_display,
        "dl_sum": dl_sum,
        "dl_model": dl_model,
        "dl_training": dl_training,
        "dl_metrics": dl_metrics,
    }
