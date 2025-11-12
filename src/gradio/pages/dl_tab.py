from __future__ import annotations
import gradio as gr

def render_dl_tab(app_desc_dl: str) -> None:
    """Onglet Deep Learning (placeholder)."""
    with gr.Tab(f"Deep Learning - {app_desc_dl}"):
        gr.Markdown("## À venir\nInterface dédiée aux modèles DL (CNN/RNN/Transformers).")
