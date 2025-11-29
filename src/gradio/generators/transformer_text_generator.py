import re
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TransformerTextGenerator:
    """
    Générateur de texte pour le modèle Transformer Keras.

    - Charge le corpus texte depuis PROJECT_ROOT / data
    - Nettoie le texte comme dans le notebook
    - Reconstruit le tokenizer (word-level)
    - Génère du texte en mode auto-régressif
    """

    # Singleton interne
    _instance: "TransformerTextGenerator | None" = None

    # Références au projet / corpus
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CORPUS_DIR = (PROJECT_ROOT / "data" / "raw" / "nlp")
    CORPUS_LENGTH_PARAM = "_car_FULL_"  # même filtre que dans le notebook FULL_50

    def __init__(self, model, tokenizer, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Méthodes de classe : singleton / factory
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, model) -> "TransformerTextGenerator":
        """
        Retourne une unique instance, basée sur :
        - le modèle Transformer déjà chargé
        - le corpus texte dans PROJECT_ROOT / data
        """
        if cls._instance is not None:
            return cls._instance

        full_text_clean = cls._load_full_clean_corpus()
        tokenizer = cls._build_tokenizer_from_corpus(full_text_clean)

        # Dans le notebook : Input(shape=(max_length - 1,))
        max_length = model.input_shape[1] + 1

        cls._instance = cls(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return cls._instance

    # ------------------------------------------------------------------
    # Chargement du corpus & nettoyage
    # ------------------------------------------------------------------
    @classmethod
    def _load_full_clean_corpus(cls) -> str:
        """
        Reproduit la logique du notebook :

        - charge les .txt dans PROJECT_ROOT / "data"
        - filtre avec CORPUS_LENGTH_PARAM
        - concatène
        - applique clean_text(...)
        """
        corpus_dir = cls.CORPUS_DIR
        if not corpus_dir.exists():
            raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")

        corpus_paths = sorted(
            p
            for p in corpus_dir.glob("*.txt")
            if (not cls.CORPUS_LENGTH_PARAM or cls.CORPUS_LENGTH_PARAM in p.name)
        )

        if not corpus_paths:
            raise FileNotFoundError(
                f"No corpus .txt files found in {corpus_dir} "
                f"(filter='{cls.CORPUS_LENGTH_PARAM}' )."
            )

        corpus_texts = []
        for path in corpus_paths:
            with open(path, encoding="utf-8") as f:
                corpus_texts.append(f.read())

        full_text = "\n".join(text.strip() for text in corpus_texts)
        full_text_clean = cls.clean_text(full_text)
        return full_text_clean

    @staticmethod
    def clean_text(texte: str) -> str:
        """Nettoie et normalise le texte (version notebook)."""
        texte = texte.lower()
        texte = texte.replace("'", "")
        texte = re.sub(r"([.,!?])", r" \1 ", texte)
        texte = re.sub(r"\s+", " ", texte)
        return texte.strip()

    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    @staticmethod
    def _build_tokenizer_from_corpus(full_text_clean: str) -> Tokenizer:
        """
        Recrée le tokenizer EXACTEMENT comme dans le notebook :

            tokenizer = Tokenizer(filters='', lower=False, oov_token='<UNK>')
            tokenizer.fit_on_texts([full_text_clean])
        """
        tok = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tok.fit_on_texts([full_text_clean])
        return tok

    def _encode_prompt(self, prompt: str) -> list[int]:
        """Nettoie le prompt et le convertit en liste d’IDs tokens."""
        prompt_clean = self.clean_text(prompt)
        token_list = self.tokenizer.texts_to_sequences([prompt_clean])[0]
        return token_list

    # ------------------------------------------------------------------
    # Génération
    # ------------------------------------------------------------------
    def generate_text(
        self,
        seed_text: str,
        num_words: int = 60,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        """
        Génère du texte à partir d’un prompt initial, en mode auto-régressif.

        - temperature contrôle la créativité
        - num_words = nombre de tokens supplémentaires à générer
        """
        seed_text = seed_text.strip()
        if not seed_text:
            return ""

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random

        generated_text = seed_text
        token_list = self._encode_prompt(seed_text)

        for _ in range(num_words):
            # Padding / tronquage à max_length-1 comme au training
            sequence = pad_sequences(
                [token_list],
                maxlen=self.max_length - 1,
                padding="pre",
            )

            # Prédiction du prochain token
            preds = self.model.predict(sequence, verbose=0)[0]

            # Température
            preds = np.log(preds + 1e-7) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))

            # Échantillonnage
            next_id = rng.choice(len(preds), p=preds)

            # Décodage ID -> mot
            word = self.tokenizer.index_word.get(next_id, "")

            if not word:
                continue

            generated_text += " " + word
            token_list.append(next_id)

        return generated_text
