import os
from src.gradio.app import build_app

if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))