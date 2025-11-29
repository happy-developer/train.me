import torch

GPT2_FINE_TUNING_GENERATOR: "GPT2_FineTuningTextGenerator | None" = None


class GPT2_FineTuningTextGenerator:
    """
    Wrapper simple autour du modèle GPT-2 fine-tuné de TrAIn.me.
    Gère :
    - la génération autoregressive
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
        temperature: float = 0.9,
        top_p: float = 0.95,
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
