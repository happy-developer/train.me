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
    goal_state: gr.State
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

        # Champ Prompt (auto-généré, non éditable)
        prompt_box = gr.Textbox(
            label="Prompt sent to the Deep Learning model",
            interactive=True,
            lines=4,
        )

        # Sélecteur du modèle DL
        model_selector = gr.Dropdown(
            label="Deep Learning model",
            choices=[
                "xMas - GPT2 Fine-tuning",
            ],
            value="xMas - GPT2 Fine-tuning",
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

        def _build_prompt(level_text: str, wf_text: str, wt_text: str, program_row, goal_text: str,):
            """
            Construit le prompt à partir du profil + programme sélectionné.
            """
            level = level_text or "Unknown"
            wf = wf_text or "N/A"
            wt = wt_text or "Olympic Weightlifting"
            goal = goal_text or "Olympic Weightlifting"

            if not program_row:
                # Aucun programme sélectionné encore
                return (
                    "Generate a workout program based on the user's physical level, "
                    "and your goal. No specific exercise has "
                    "been selected yet."
                )

            ex_name = program_row.get("exercise_name") or "the selected exercise"
            target = program_row.get("target_muscles") or "the target muscles"
            equip = program_row.get("equipment") or "bodyweight only"

            # Prompt au format demandé
            prompt = (
                f"{level} level ({goal})\n\n"
            )
            return prompt

        # --- Callbacks ---

        def _on_dl_model_select(name: str):
            status = on_model_change(name)
            df_sum, df_model, df_training, df_metrics = get_dl_model_report_components(
                name
            )
            return status, df_sum, df_model, df_training, df_metrics

        model_selector.change(
            fn=_on_dl_model_select,
            inputs=model_selector,
            outputs=[model_status, dl_sum, dl_model, dl_training, dl_metrics],
        )

        # Synchronisation du profil + programme + prompt à l'ouverture du tab DL
        def _sync_profile(level_val, wf_val, wt_val, program_row, goal_val):
            level_text = level_val or ""
            wf_text = "" if wf_val in (None, "") else str(wf_val)
            wt_text = wt_val or ""
            goal_text = goal_val or "Olympic Weightlifting"

            if not program_row:
                program_df = pd.DataFrame()
            else:
                program_df = pd.DataFrame([program_row])

            prompt = _build_prompt(level_text, wf_text, wt_text, program_row, goal_text)

            return (
                level_text,
                wf_text,
                wt_text,
                goal_text,
                program_df,
                prompt,
            )

        tab_dl.select(
            _sync_profile,
            inputs=[level_out, wf_comp, wt_comp,  selected_program_df, goal_state],
            outputs=[level_display, wf_display, wt_display, goal_display, program_display, prompt_box],
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
    }
