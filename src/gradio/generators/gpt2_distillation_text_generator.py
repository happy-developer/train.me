import textwrap
import torch


class GPT2_DistilledTextGenerator:
    """
    Wrapper simple pour le modèle GPT-2 distillé de TrAIn.me.
    Gère :
    - la génération auto-régressive
    - temperature / top_p
    - max_new_tokens
    """

    def __init__(self, model, tokenizer, max_new_tokens: int = 256):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def generate_text(
        self,
        prompt: str,
        temperature: float = 0.8,
        top_p: float = 0.9,
        strip_prompt: bool = True,
    ) -> str:
        """Génère du texte à partir du prompt."""
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


    def generer_exercice_interactif(
        self,
        workout_type: str = "strength",
        debut: str = "",
        num_samples: int = 3,
        max_length: int = 120,
        temperature: float = 1.0,
    ):
        """
        Génère plusieurs descriptions / consignes d’entraînement
        avec le modèle Student distillé (GPT-2 compact spécialisé TrAIn.me).

        Args:
            model : modèle HuggingFace (AutoModelForCausalLM)
            tokenizer : tokenizer associé (teacher_tokenizer recommandé)
            workout_type : catégorie (strength, cardio, mobility, core…)
            debut : début de phrase fourni par l'utilisateur
            num_samples : nombre d’exemples à générer
            max_length : longueur maximale en tokens
            temperature : contrôle la créativité (>1 = plus créatif)
        """
        device = torch.device("cpu")
        # Construction du prompt
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

            # Décodage
            generated_text = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True
            )

            # print(f"\n> Exemple #{i+1} :")
            return(
                textwrap.fill(
                    generated_text,
                    width=75,
                    initial_indent="   ",
                    subsequent_indent="   ",
                )
            )
            # print("-" * 80)