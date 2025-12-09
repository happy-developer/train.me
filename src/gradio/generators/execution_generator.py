from pathlib import Path
from functools import lru_cache
from typing import Mapping, Union, List

import re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------------------------------------------------
# Paths + device
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "v1"

# Dossier de ton modèle finetuné d'exécution
# EXEC_MODEL_DIR = MODEL_DIR / "transformer_execution_generator_v3"
# Chargement du modèle HF (tokenizer + modèle)
MODEL_REPO = "AIppyDev/transformer_execution_generator_v3"
MODEL_SUBFOLDER = "transformer_execution_generator_v3"  # le nom du dossier dans le repo

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Nombre max de tokens générés (équivalent MAX_LENGTH du notebook)
EXEC_MAX_NEW_TOKENS = 180


# ---------------------------------------------------------------------
# Construction du prompt pour un programme
# ---------------------------------------------------------------------

def build_execution_prompt(row: Union[pd.Series, Mapping, None]) -> str:
    """
    Construit le prompt d'exécution à partir d'une ligne de programme.

    Format :
    The {exercise_name} exercise for "{target_muscles}" with "{equipment}" on "{difficulty}" level:
    """
    if row is None:
        return ""

    def get_val(key: str, default: str = "") -> str:
        if isinstance(row, pd.Series):
            val = row.get(key, default)
        else:
            val = row.get(key, default) if hasattr(row, "get") else default

        if pd.isna(val):
            return default
        return str(val).strip()

    name = get_val("exercise_name", "this exercise")
    muscle = get_val("target_muscles", "the target muscles")
    equipment = get_val("equipment", "bodyweight")
    difficulty = get_val("difficulty", "General")

    prompt = (
        f'The {name} exercise for '
        f'"{muscle}" with '
        f'"{equipment}" on '
        f'"{difficulty}" level:'
    )
    return prompt


# ---------------------------------------------------------------------
# Utils de post-traitement
# ---------------------------------------------------------------------

def strip_prompt_from_generations(prompts: List[str], generated_outputs: List[str]) -> List[str]:
    """
    Supprime le prompt au début de chaque génération si le modèle l'a recopié.
    """
    cleaned = []

    for prompt, gen in zip(prompts, generated_outputs):
        gen_strip = gen.strip()

        if gen_strip.startswith(prompt):
            cleaned.append(gen_strip[len(prompt):].strip())
        else:
            cleaned.append(gen_strip)

    return cleaned


def trim_to_last_full_sentence(text: str) -> str:
    """
    Coupe le texte à la dernière phrase complète.
    On garde tout jusqu'au dernier '.', '!' ou '?'.
    Si aucun n'est trouvé, on renvoie le texte brut stripé.
    """
    text = text.strip()

    last_dot = text.rfind(".")
    last_excl = text.rfind("!")
    last_q = text.rfind("?")

    last_punct = max(last_dot, last_excl, last_q)

    if last_punct != -1:
        return text[: last_punct + 1].strip()
    return text


def keep_first_n_sentences(text: str, n: int = 2) -> str:
    """
    Garde seulement les n premières phrases (séparation grossière sur . ! ?).
    """
    sentences = re.split(r'([.!?])', text)
    chunks = []
    count = 0

    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        punct = sentences[i + 1].strip()
        if sentence:
            chunks.append(sentence + punct)
            count += 1
        if count >= n:
            break

    return " ".join(chunks).strip() if chunks else text.strip()


def dedupe_muscle_list(text: str) -> str:
    """
    Détecte les listes du type 'glutes, hamstrings, and hamstrings'
    dans la phrase, les déduplique et les reformate proprement.
    """

    # Regex qui capture une liste séparée par virgules ou 'and'
    # Exemple capturé : "glutes, hamstrings, and hamstrings"
    pattern = r"([A-Za-z ]+(?:,\s*[A-Za-z ]+)*(?:\s+and\s+[A-Za-z ]+)?)"

    def process_match(match):
        segment = match.group(0)

        # Split sur virgules et 'and'
        parts = re.split(r",|\band\b", segment)
        parts = [p.strip() for p in parts if p.strip()]

        # Déduplique tout en gardant l’ordre
        seen = set()
        unique = []
        for p in parts:
            if p.lower() not in seen:
                seen.add(p.lower())
                unique.append(p)

        # Reconstruction naturelle
        if len(unique) == 1:
            return unique[0]

        if len(unique) == 2:
            return f"{unique[0]} and {unique[1]}"

        return ", ".join(unique[:-1]) + f", and {unique[-1]}"

    # Applique la fonction sur toutes les occurrences
    cleaned = re.sub(pattern, process_match, text)
    return cleaned


def clean_execution_text(text: str) -> str:
    """
    Nettoyage final : espaces, 'reps', etc.
    """
    if not isinstance(text, str):
        return text

    cleaned = text

    # 1) Ajouter un espace après un point si collé à une majuscule
    cleaned = re.sub(r'(\.)([A-Z])', r'\1 \2', cleaned)

    # 2) Ajouter un espace après "such as" si collé à un chiffre
    cleaned = re.sub(r"(such as)(\d)", r"\1 \2", cleaned)

    # 3) Ajouter un espace avant "reps" si collé au chiffre
    cleaned = re.sub(r"(\d)(reps)", r"\1 reps", cleaned)

    # 4) Normaliser les doubles espaces
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return cleaned


# ---------------------------------------------------------------------
# Chargement du modèle HF (tokenizer + modèle)
# ---------------------------------------------------------------------

@lru_cache()
def _load_exec_model():
    """
    Charge une seule fois tokenizer + modèle pour l'execution generator.
    Utilise un cache pour éviter les rechargements coûteux.
    """

    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO,
            subfolder=MODEL_SUBFOLDER,)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
            subfolder=MODEL_SUBFOLDER,
        torch_dtype=torch.float32,   # CPU friendly
    )
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------
# Génération de texte pour l'exécution (1 prompt, pipeline complet)
# ---------------------------------------------------------------------

def generate_execution_text(
    prompt: str,
    max_new_tokens: int = EXEC_MAX_NEW_TOKENS,
    temperature: float = 0.8,
    top_k: int = 250,
    top_p: float = 0.92,
) -> str:
    """
    Génère une description d'exécution à partir d'un prompt unique,
    avec le même pipeline que dans le notebook d'entraînement.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return "No prompt available to generate execution."

    tokenizer, model = _load_exec_model()

    sample_prompts = [prompt]

    encodings = tokenizer(
        sample_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    # 1. Génération
    outputs = model.generate(
        **encodings,
        max_new_tokens=max_new_tokens,   # nombre maximum de tokens générés
        min_new_tokens=50,               # longueur minimale
        do_sample=True,                  # sampling activé (obligatoire pour top-k / top-p)
        temperature=temperature,         # légère "chauffe" pour plus de variété
        top_k=top_k,                     # top-K sampling large
        top_p=top_p,                     # nucleus sampling
        no_repeat_ngram_size=3,          # évite les répétitions de 3-grams
        num_beams=5,                     # beam search pour améliorer la cohérence
        num_return_sequences=1,          # une seule génération par prompt
        pad_token_id=tokenizer.eos_token_id,
        early_stopping=True,            # arrêt anticipé si EOS atteint partout
    )

    generated_outputs = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True,
    )

    # 2. On enlève le prompt au début de chaque génération
    cleaned_generations = strip_prompt_from_generations(
        prompts=sample_prompts,
        generated_outputs=generated_outputs,
    )

    # 3. Post-traitement élément par élément
    trimmed = [trim_to_last_full_sentence(txt) for txt in cleaned_generations]
    limited = [keep_first_n_sentences(t, n=2) for t in trimmed]
    deduped = [dedupe_muscle_list(t) for t in limited]
    final = [clean_execution_text(t) for t in deduped]

    return final[0] if final else ""
