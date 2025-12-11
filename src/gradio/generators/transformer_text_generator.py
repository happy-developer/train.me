import re
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Mapping

import numpy as np
import torch
import torch.nn as nn

import tensorflow as tf
from tensorflow.keras.utils import register_keras_serializable

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model as keras_load_model


# ======================================================================
#  BLOCS PYTORCH : doivent matcher ceux du notebook transformer_trainme_v2
# ======================================================================


class TorchPositionalEmbedding(nn.Module):
    """
    Version PyTorch de l'embedding de tokens + encodage positionnel,
    utilisée pour le Transformer v2 (.pt).
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_length: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.max_length = max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len)  →  (batch, seq_len, embed_dim)
        (dans le notebook v2, le positional encoding était implicite / fixe)
        """
        token_embeddings = self.token_emb(x)
        return token_embeddings


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention maison, comme dans le notebook v2.

    - query, key, value : Linear(embed_dim → embed_dim)
    - out : Linear(embed_dim → embed_dim)
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout_rate: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout_rate)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        q, k, v : (batch, num_heads, seq_len, head_dim)
        mask    : (seq_len, seq_len) ou (batch, 1, seq_len, seq_len)
        """
        dk = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(dk)  # (b, h, L, L)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        output = torch.matmul(attn_weights, v)  # (b, h, L, head_dim)
        return output

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x : (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.size()

        # Projections linéaires
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Split en têtes : (b, L, h, d_h) → (b, h, L, d_h)
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention
        attn_output = self._scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads
        attn_output = (
            attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        )

        # Projection finale
        out = self.out(attn_output)
        return out


class TransformerBlock(nn.Module):
    """
    Bloc Transformer standard :
    - MultiHeadSelfAttention
    - Add & Norm
    - FFN (Linear → ReLU → Linear)
    - Add & Norm
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads, dropout_rate=dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )

        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.dropout_att = nn.Dropout(dropout_rate)
        self.dropout_ffn = nn.Dropout(dropout_rate)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Masque triangulaire inférieur (causal) de taille (1, 1, seq_len, seq_len).
        """
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        device = x.device

        mask = self._causal_mask(seq_len, device)

        # Self-attention + residual
        attn_output = self.att(x, mask=mask)
        x = self.layernorm1(x + self.dropout_att(attn_output))

        # FFN + residual
        ffn_output = self.ffn(x)
        x = self.layernorm2(x + self.dropout_ffn(ffn_output))

        return x


class TransformerLanguageModel(nn.Module):
    """
    Modèle PyTorch complet pour la génération de texte (version du notebook v2).

    Input tokens →
        TorchPositionalEmbedding →
        [TransformerBlock] × N →
        Dropout →
        Linear(embed_dim → vocab_size) sur la DERNIÈRE position
    """

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        num_blocks: int,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate

        self.embedding = TorchPositionalEmbedding(vocab_size, embed_dim, max_length)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout_rate=dropout_rate,
                )
                for _ in range(num_blocks)
            ]
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len)  →  logits : (batch, vocab_size) sur la dernière position.
        """
        x = self.embedding(x)

        for block in self.blocks:
            x = block(x)

        x = self.dropout(x)

        last_token = x[:, -1, :]
        logits = self.fc_out(last_token)
        return logits


# ======================================================================
#  BLOC KERAS : PositionalEmbedding pour recharger Transformer_v1 (.keras)
# ======================================================================


@register_keras_serializable(package="Custom", name="PositionalEmbedding")
class PositionalEmbedding(tf.keras.layers.Layer):
    """
    Version Keras de PositionalEmbedding, pour recharger le modèle v1
    `transformer_wordlevel_v1.keras` qui a été sauvegardé avec cette couche
    custom dans sa config.

    Les champs attendus dans la config sont :
    - vocab_size
    - embed_dim
    - max_len
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_len: int = 1024, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_len, embed_dim)

    def call(self, x):
        # x : (batch, seq_len)
        length = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        positions = self.pos_emb(positions)
        token_embeddings = self.token_emb(x)
        return token_embeddings + positions

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
                "max_len": self.max_len,
            }
        )
        return config


# ======================================================================
#  GÉNÉRATEUR DE TEXTE : TransformerTextGenerator
# ======================================================================


class TransformerTextGenerator:
    """
    Générateur de texte pour le modèle Transformer (v1 Keras ou v2 PyTorch).

    - Charge le corpus texte depuis PROJECT_ROOT / gradio/data/*.txt
    - Nettoie le texte comme dans le notebook
    - Reconstruit le tokenizer (word-level)
    - Charge soit :
        * un modèle Keras (.keras) avec PositionalEmbedding custom
        * un modèle PyTorch (.pt) reconstruit via TransformerLanguageModel
    """

    _instance: "TransformerTextGenerator | None" = None

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CORPUS_DIR = PROJECT_ROOT / "gradio" / "data"
    CORPUS_LENGTH_PARAM = "_car_FULL_"

    DEFAULT_MAX_LENGTH = 50

    def __init__(self, model, tokenizer: Tokenizer, max_length: int):
        # On veut un objet modèle (Keras ou PyTorch), pas un state_dict
        if isinstance(model, (dict, OrderedDict)):
            raise TypeError(
                "TransformerTextGenerator doit recevoir un modèle déjà construit, "
                "pas un state_dict brut."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        self._is_torch = isinstance(self.model, nn.Module)
        if self._is_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        else:
            self.device = None

    # ------------------------------------------------------------------
    # Chargement depuis un chemin (Keras .keras ou PyTorch .pt)
    # ------------------------------------------------------------------
    @staticmethod
    def _load_model_from_path(path: Union[str, Path]):
        path = Path(path)
        print(f"[TransformerTextGenerator] Loading model from path: {path}")

        suffix = path.suffix.lower()

        if suffix == ".keras":
            # v1 : modèle Keras avec couche PositionalEmbedding custom
            return keras_load_model(
                path,
                compile=False,
                custom_objects={"PositionalEmbedding": PositionalEmbedding},
            )

        if suffix == ".pt":
            # v2 : checkpoint PyTorch
            state = torch.load(path, map_location="cpu")
            if isinstance(state, Mapping):
                # Reconstruction du TransformerLanguageModel
                return TransformerTextGenerator._build_torch_model_from_state_dict(state)
            elif isinstance(state, nn.Module):
                return state
            else:
                raise TypeError(
                    f"Contenu inattendu dans le checkpoint PyTorch {path}: {type(state)}"
                )

        raise ValueError(f"Unsupported model file extension for {path}")

    # ------------------------------------------------------------------
    # Reconstruction du modèle PyTorch depuis un state_dict
    # ------------------------------------------------------------------
    @classmethod
    def _build_torch_model_from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
    ) -> TransformerLanguageModel:
        """
        Reconstruit un `TransformerLanguageModel` à partir d’un state_dict
        tel que sauvegardé dans le notebook transformer_trainme_v2.
        """

        if "embedding.token_emb.weight" not in state:
            raise KeyError(
                "Le state_dict fourni ne contient pas la clé "
                "'embedding.token_emb.weight'. "
                "Vérifie que tu utilises bien le checkpoint du TransformerLanguageModel v2."
            )

        token_emb_weight = state["embedding.token_emb.weight"]
        vocab_size = token_emb_weight.shape[0]
        embed_dim = token_emb_weight.shape[1]

        ffn0_key = "blocks.0.ffn.0.weight"
        if ffn0_key not in state:
            raise KeyError(
                f"Clé '{ffn0_key}' absente du state_dict. "
                "Structure attendue : blocks.{i}.ffn.0.weight."
            )
        ff_dim = state[ffn0_key].shape[0]

        num_blocks = 0
        while f"blocks.{num_blocks}.att.query.weight" in state:
            num_blocks += 1
        if num_blocks == 0:
            raise ValueError(
                "Impossible de déterminer le nombre de blocs Transformer "
                "(aucune clé 'blocks.{i}.att.query.weight' trouvée)."
            )

        num_heads = 4
        max_length = cls.DEFAULT_MAX_LENGTH
        dropout_rate = 0.1

        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        )
        model.load_state_dict(state)
        return model

    # ------------------------------------------------------------------
    # Singleton / factory
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, model_or_state_or_path) -> "TransformerTextGenerator":
        """
        Retourne une unique instance, basée sur :
        - un chemin vers un .keras ou .pt
        - un state_dict (OrderedDict) PyTorch
        - ou un nn.Module déjà construit.
        """

        if cls._instance is not None:
            return cls._instance

        # 1) Normalisation de l'entrée
        if isinstance(model_or_state_or_path, (str, Path)):
            model = cls._load_model_from_path(model_or_state_or_path)
        else:
            model = model_or_state_or_path

        # Si on reçoit un state_dict brut → reconstruire le modèle PyTorch
        if isinstance(model, Mapping):
            model = cls._build_torch_model_from_state_dict(model)

        # 2) Charger + nettoyer le corpus, rebuild du tokenizer
        full_text_clean = cls._load_full_clean_corpus()
        tokenizer = cls._build_tokenizer_from_corpus(full_text_clean)

        # 3) max_length : si le modèle l'expose, on le récupère
        max_length = getattr(model, "max_length", cls.DEFAULT_MAX_LENGTH)

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
                f"(filter='{cls.CORPUS_LENGTH_PARAM}')"
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
        tok = Tokenizer(filters="", lower=False, oov_token="<UNK>")
        tok.fit_on_texts([full_text_clean])
        return tok

    def _encode_prompt(self, prompt: str) -> list[int]:
        prompt_clean = self.clean_text(prompt)
        token_list = self.tokenizer.texts_to_sequences([prompt_clean])[0]
        return token_list

    # ------------------------------------------------------------------
    # Backend-unified prediction
    # ------------------------------------------------------------------
    def _predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Unifie la prédiction entre Keras (.predict) et PyTorch (.forward).
        """
        if self._is_torch:
            x = torch.from_numpy(sequence).long().to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                if logits.dim() == 2:
                    logits = logits[0]
                else:
                    logits = logits.view(-1)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        else:
            # Modèle Keras (Transformer v1)
            preds = self.model.predict(sequence, verbose=0)
            if isinstance(preds, (list, tuple)):
                preds = preds[0]
            preds = np.asarray(preds)
            if preds.ndim == 2:
                return preds[0]
            return preds

    # ------------------------------------------------------------------
    # Génération
    # ------------------------------------------------------------------
    def generate_text(
        self,
        seed_text: str,
        num_words: int = 60,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> str:
        seed_text = seed_text.strip()
        if not seed_text:
            return ""

        rng = np.random.default_rng(seed) if seed is not None else np.random

        generated_text = seed_text
        token_list = self._encode_prompt(seed_text)

        for _ in range(num_words):
            if not token_list:
                break

            sequence = pad_sequences(
                [token_list],
                maxlen=self.max_length - 1,
                padding="pre",
            )

            preds = self._predict_proba(sequence.astype("int64"))

            preds = np.log(preds + 1e-7) / temperature
            preds = np.exp(preds) / np.sum(np.exp(preds))

            next_id = rng.choice(len(preds), p=preds)
            word = self.tokenizer.index_word.get(next_id, "")

            if not word:
                continue

            generated_text += " " + word
            token_list.append(next_id)

        return generated_text
