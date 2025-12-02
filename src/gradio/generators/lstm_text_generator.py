import re
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import pandas as pd  # <-- NEW

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class LSTMTextGenerator:
    """
    Gestionnaire dédié pour le LSTM :
    - reçoit un modèle déjà chargé (PyTorch ou Keras) depuis on_model_change
    - OU un state_dict PyTorch (OrderedDict) → reconstruit alors un nn.Module
    - charge + nettoie le corpus texte
    - recrée le tokenizer (identique au notebook LSTM v3)
    - génère du texte à partir d'un seed.
    """

    # Racine du projet + chemin vers le corpus CSV
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CORPUS_CSV_PATH = PROJECT_ROOT / "gradio" / "nlp" / "program_summary.csv"

    # Colonnes textuelles utilisées dans le notebook LSTM v3
    TEXT_COLUMNS = [
        "program_title",
        "description",
        "goal",
        "target_muscles",
        "equipment",
        "instructions",
    ]

    # Longueur de séquence par défaut (cf. LSTM_model_report.json)
    DEFAULT_SEQUENCE_LENGTH = 15

    # Singleton interne
    _instance: "LSTMTextGenerator | None" = None

    # ----------------------------------------------------------------------
    # Constructeur
    # ----------------------------------------------------------------------
    def __init__(self, model, tokenizer, max_length: int | None = None):

        # Si on reçoit un state_dict PyTorch → reconstruire un nn.Module
        if isinstance(model, OrderedDict):
            model = self._build_torch_model_from_state_dict(model)

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length or self.DEFAULT_SEQUENCE_LENGTH

        # Backend PyTorch ou Keras
        self._is_torch = isinstance(self.model, torch.nn.Module)
        if self._is_torch:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(self.device)
            self.model.eval()
        else:
            self.device = None

    # ----------------------------------------------------------------------
    # Reconstruction d'un modèle PyTorch à partir d'un state_dict
    # ----------------------------------------------------------------------
    @staticmethod
    def _build_torch_model_from_state_dict(state: OrderedDict) -> torch.nn.Module:
        """
        Reconstruit dynamiquement un LSTM language model standard à partir d'un state_dict
        de la forme :

            embedding.weight  -> (vocab_size, embedding_dim)
            lstm.weight_ih_l0 -> (4*hidden_dim, embedding_dim)
            fc.weight         -> (vocab_size, hidden_dim)
        """

        emb_weight = state.get("embedding.weight", None)
        fc_weight = state.get("fc.weight", None)
        lstm_weight_ih_l0 = state.get("lstm.weight_ih_l0", None)

        if emb_weight is None or fc_weight is None or lstm_weight_ih_l0 is None:
            raise TypeError(
                "Impossible de reconstruire le LSTM PyTorch : "
                "les clés attendues 'embedding.weight', 'lstm.weight_ih_l0', "
                "'fc.weight' sont absentes du state_dict. "
            )

        vocab_size, embedding_dim = emb_weight.shape
        out_features, hidden_dim = fc_weight.shape

        if out_features != vocab_size:
            print(
                f"[LSTMTextGenerator] Avertissement : fc.weight shape={fc_weight.shape} "
                f"→ out_features != vocab_size ({out_features} != {vocab_size})."
            )

        num_layers = 1
        for n in range(1, 10):
            if f"lstm.weight_ih_l{n}" in state:
                num_layers = n + 1
            else:
                break

        class TorchLSTMLanguageModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim)
                self.lstm = nn.LSTM(
                    input_size=embedding_dim,
                    hidden_size=hidden_dim,
                    num_layers=num_layers,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_dim, vocab_size)

            def forward(self, x):
                emb = self.embedding(x)          # (batch, seq_len, embed_dim)
                out, _ = self.lstm(emb)          # (batch, seq_len, hidden_dim)
                logits = self.fc(out)            # (batch, seq_len, vocab_size)
                return logits

        model = TorchLSTMLanguageModel(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )
        model.load_state_dict(state)
        return model

    # ----------------------------------------------------------------------
    # Méthodes de classe : singleton
    # ----------------------------------------------------------------------
    @classmethod
    def get_instance(cls, model) -> "LSTMTextGenerator":
        """
        Retourne une unique instance de LSTMTextGenerator, initialisée à partir :
        - d'un modèle LSTM déjà chargé (Keras ou PyTorch)
        - ou d'un state_dict PyTorch (OrderedDict)
        - du corpus texte dans data/programs/program_summary.csv
        """
        if cls._instance is not None:
            return cls._instance

        # Charger + nettoyer le corpus (version CSV anglaise)
        full_text_clean = cls._load_full_clean_corpus()

        # Recréer le tokenizer EXACT comme dans le notebook
        tokenizer = cls.build_tokenizer_from_corpus(full_text_clean)

        max_length = cls.DEFAULT_SEQUENCE_LENGTH
        if hasattr(model, "input_shape") and getattr(model, "input_shape") is not None:
            try:
                max_length = int(model.input_shape[1]) + 1
            except Exception:
                pass

        cls._instance = cls(
            model=model,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return cls._instance

    # ----------------------------------------------------------------------
    # Helpers internes : corpus + nettoyage + tokenizer
    # ----------------------------------------------------------------------
    @classmethod
    def _load_full_clean_corpus(cls) -> str:
        """
        Version LSTM v3 :
        - charge `program_summary.csv`
        - concatène plusieurs colonnes textuelles
        - applique clean_text(...)
        """
        csv_path = cls.CORPUS_CSV_PATH
        if not csv_path.exists():
            raise FileNotFoundError(f"Corpus CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        # On garde seulement les colonnes réellement présentes
        text_cols = [c for c in cls.TEXT_COLUMNS if c in df.columns]
        if not text_cols:
            raise ValueError(
                f"Aucune des colonnes textuelles attendues {cls.TEXT_COLUMNS} "
                f"n'a été trouvée dans {csv_path.name}."
            )

        df[text_cols] = df[text_cols].fillna("")

        # Concaténation ligne par ligne (comme dans le notebook)
        lines = []
        for _, row in df[text_cols].iterrows():
            line = " ".join(str(row[c]) for c in text_cols).strip()
            if line:
                lines.append(line)

        full_text = "\n".join(lines)
        return cls.clean_text(full_text)

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
        tok = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tok.fit_on_texts([full_text_clean])
        return tok

    # ----------------------------------------------------------------------
    # Prédiction : unification Keras / PyTorch
    # ----------------------------------------------------------------------
    def _predict_proba(self, token_list: np.ndarray) -> np.ndarray:
        if self._is_torch:
            x = torch.from_numpy(token_list).long().to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                if logits.dim() == 3:
                    logits = logits[:, -1, :]
                logits = logits[0]
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        else:
            preds = self.model.predict(token_list, verbose=0)
            return preds[0]

    # ----------------------------------------------------------------------
    # Génération auto-régressive
    # ----------------------------------------------------------------------
    def generate_text(
        self,
        seed_text: str,
        num_words: int = 40,
        temperature: float = 0.8,
        seed: int | None = None,
    ) -> str:
        rng = np.random.default_rng(seed) if seed is not None else np.random
        generated_text = seed_text

        for _ in range(num_words):
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
            token_list = pad_sequences(
                [token_list],
                maxlen=self.max_length - 1,
                padding="pre",
            )

            predictions = self._predict_proba(token_list)
            predictions = np.log(predictions + 1e-7) / temperature
            predictions = np.exp(predictions) / np.sum(np.exp(predictions))

            predicted_id = rng.choice(len(predictions), p=predictions)

            predicted_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_id:
                    predicted_word = word
                    break

            if predicted_word:
                generated_text += " " + predicted_word

        return generated_text
