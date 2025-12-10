from pathlib import Path
import textwrap
import torch
import re

from transformers import AutoModelForCausalLM, AutoTokenizer

class GPT2_DistilledTextGenerator:
    _INSTANCES = {}

    @classmethod
    def get_instance(
        cls,
        model_path,
        max_new_tokens: int = 256
    ) -> "GPT2_DistilledTextGenerator":
        """
        Charge (ou récupère en cache) un générateur GPT-2 distillé
        depuis un dossier `model_path` (local ou NAS).
        """
        model_path = Path(model_path)

        # On log pour debug
        print(f"[GPT2_Distilled] Loading from: {model_path}")

        # IMPORTANT : ne plus lever FileNotFoundError ici,
        # on laisse Transformers gueuler si le dossier est vraiment faux.

        local_dir = model_path.as_posix()

        # cache par chemin string
        if local_dir in cls._INSTANCES:
            return cls._INSTANCES[local_dir]

        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            local_files_only=True,
        )

        instance = cls(model=model, tokenizer=tokenizer, max_new_tokens=max_new_tokens)
        cls._INSTANCES[local_dir] = instance
        return instance

    def __init__(self, model, tokenizer, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Génération simple
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        strip_prompt: bool = True,
    ) -> str:

        prompt = prompt.strip()
        if not prompt:
            return "Please enter a prompt before generating."

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        if strip_prompt and text.startswith(prompt):
            text = text[len(prompt):].lstrip()

        return text

    # ------------------------------------------------------------------
    # Ancienne fonction interactive (simple)
    # ------------------------------------------------------------------
    def generer_exercice_interactif(
        self,
        workout_type: str = "strength",
        debut: str = "",
        num_samples: int = 3,
        max_length: int = 120,
        temperature: float = 1.0,
    ):
        device = torch.device("cpu")

        if debut:
            prompt = f"Workout [{workout_type}]: {debut}"
        else:
            prompt = f"Workout [{workout_type}]:"

        print(f"\nGénération de {num_samples} exemples ({workout_type.upper()})")
        print(f"Prompt : '{prompt}'\n")
        print("=" * 80)

        self.model.to(device)
        self.model.eval()

        for i in range(num_samples):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            return textwrap.fill(
                generated_text,
                width=75,
                initial_indent="   ",
                subsequent_indent="   ",
            )


    # ------------------------------------------------------------------
    # Filtre amélioré
    # ------------------------------------------------------------------
    def generate_with_filter(
        self,
        model,
        tokenizer,
        prompt: str,
        goal: str,
        n_candidates: int = 3,
        max_new_tokens: int = 160,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        max_attempts: int = 3,
    ):
        device = torch.device("cpu")
        model.to(device)
        model.eval()

        self.model.to(device)
        self.model.eval()

        for i in range(1):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_length=160,
                    do_sample=True,
                    temperature=temperature,
                    top_k=50,
                    top_p=0.95,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            return textwrap.fill(
                generated_text,
                width=75,
                initial_indent="   ",
                subsequent_indent="   ",
            )

    # ------------------------------------------------------------------
    # Version V2 pour l'IHM
    # ------------------------------------------------------------------



    def generer_exercice_interactif_V2(
        self,
        level: str,
        goal: str,
        num_samples: int = 3,
        max_length: int = 150,
        temperature: float = 1.0,
        candidates_per_sample: int = 3,
    ):

        prompt = f"{level} level ({goal})\n\n"

        texts = []

        for i in range(num_samples):
            # ⬇️ ICI : appel via self, plus via la classe
            generated_text = self.generate_with_filter(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                goal=goal,
                n_candidates=candidates_per_sample,
                max_new_tokens=160,
                temperature=0.7,
                top_k=40,
                top_p=0.9,
            )


            
             # Suppression des artefacts de liste Python
            cleaned = generated_text.replace("['", "")
            cleaned = cleaned.replace("']", "")
            cleaned = cleaned.replace('", "', ' ')  # parfois utilisé dans les listes

            # Supprimer les retours à la ligne trop nombreux
            cleaned = cleaned.replace("\n", " ")

            # Retirer le prompt si le modèle l'a recopié
            prompt_strip = prompt.strip()
            if prompt_strip and cleaned.startswith(prompt_strip):
                cleaned = cleaned[len(prompt_strip):].lstrip()

            # Suppression des doubles espaces
            cleaned = re.sub(r"\s{2,}", " ", cleaned)

            # Trim final
            cleaned = cleaned.strip()


            texts.append(
                textwrap.fill(
                    cleaned,
                    width=75,
                    initial_indent="   ",
                    subsequent_indent="   ",
                )
            )

        return texts

