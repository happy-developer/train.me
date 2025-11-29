import re
from pathlib import Path

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


class LSTMTextGenerator:
    """
    Gestionnaire dédié pour le LSTM :
    - chargement du modèle depuis LOADED_MODELS (via on_model_change)
    - chargement + nettoyage du corpus texte
    - création du tokenizer (identique au notebook)
    - génération de texte à partir d'un seed.

    Utilisation côté app :
        lstm_gen = LSTMTextGenerator.get_instance()
        text = lstm_gen.generate_text("mon prompt ...")
    """

    # Références au projet / corpus
    CORPUS_DIR = (Path(__file__).resolve().parents[2] / "data" / "raw" / "nlp")
    CORPUS_LENGTH_PARAM = "_car_FULL_"  # même filtre que ton notebook

    # Singleton interne
    _instance: "LSTMTextGenerator | None" = None

    def __init__(self, model, tokenizer, max_length: int):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

    # ------------------------------------------------------------------
    # Méthodes de classe : singleton / factory
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, model) -> "LSTMTextGenerator":
        """
        Retourne une unique instance de LSTMTextGenerator, initialisée à partir :
        - du modèle 'LSTM' déjà géré par LOADED_MODELS / on_model_change
        - du corpus texte dans PROJECT_ROOT / data
        """
        if cls._instance is not None:
            return cls._instance

        # Charger + nettoyer le corpus
        full_text_clean = cls._load_full_clean_corpus()

        # Recréer le tokenizer EXACT comme dans le notebook
        tokenizer = cls.build_tokenizer_from_corpus(full_text_clean)

        # Dans le notebook : Input(shape=(max_length - 1,))
        max_length = model.input_shape[1] + 1

        cls._instance = cls(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return cls._instance

    # ------------------------------------------------------------------
    # Helpers internes : corpus + nettoyage + tokenizer
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

        texts = [p.read_text(encoding="utf-8").strip() for p in corpus_paths]
        full_text = "\n".join(texts)
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

    @staticmethod
    def build_tokenizer_from_corpus(full_text_clean: str) -> Tokenizer:
        """
        Reproduit exactement le tokenizer du notebook :

            tokenizer = Tokenizer(filters='', lower=False, oov_token='<UNK>')
            tokenizer.fit_on_texts([full_text_clean])
        """
        tok = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tok.fit_on_texts([full_text_clean])
        return tok

    # ------------------------------------------------------------------
    # Génération de texte (logique notebook)
    # ------------------------------------------------------------------
    def generate_text(
        self,
        seed_text: str,
        num_words: int = 40,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        """
        Génère du texte à partir d'un texte de départ.
        Reprend la logique de `generate_text` du notebook LSTM.
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random

        generated_text = seed_text

        for _ in range(num_words):
            # Tokenisation + padding
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_length - 1,
                padding="pre",
            )

            # Prédiction du prochain mot
            predictions = self.model.predict(token_list, verbose=0)[0]

            # Température
            predictions = np.log(predictions + 1e-7) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            # Tirage d'un id
            predicted_id = rng.choice(len(predictions), p=predictions)

            # Conversion id -> mot
            predicted_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_id:
                    predicted_word = word
                    break

            if predicted_word:
                generated_text += " " + predicted_word

        return generated_text
