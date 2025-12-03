import re
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Union, Mapping

import numpy as np
import torch
import torch.nn as nn
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ======================================================================
#  BLOCS PYTORCH : doivent matcher ceux du notebook transformer_trainme_v2
# ======================================================================


class PositionalEmbedding(nn.Module):
    """
    Embedding de tokens + encodage positionnel (version légère).

    Dans le notebook v2, la seule partie entraînable côté embedding est
    `token_emb : Embedding(vocab_size, embed_dim)`.
    La partie positionnelle peut rester non-paramétrique.
    """

    def __init__(self, vocab_size: int, embed_dim: int, max_length: int):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.max_length = max_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len)  →  retourne (batch, seq_len, embed_dim)
        On se contente ici d'un token embedding + encodage positionnel implicite
        (comme dans le notebook, aucun poids supplémentaire n'était enregistré
        dans le state_dict).
        """
        # (batch, seq_len, embed_dim)
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
            # mask == 0 → -inf
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
        q = self.query(x)  # (b, L, d)
        k = self.key(x)
        v = self.value(x)

        # Split en têtes : (b, L, h, d_h) → (b, h, L, d_h)
        def split_heads(t):
            return t.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        # Attention avec masque causal éventuel
        attn_output = self._scaled_dot_product_attention(q, k, v, mask=mask)

        # Merge heads : (b, h, L, d_h) → (b, L, d)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Projection finale
        out = self.out(attn_output)  # (b, L, d)
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
        Masque triangulaire inférieur (causal) de taille (1, 1, seq_len, seq_len)
        compatible avec la forme (batch, num_heads, L, L).
        """
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).unsqueeze(0).unsqueeze(0)
        return mask  # (1, 1, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (batch, seq_len, embed_dim)
        """
        seq_len = x.size(1)
        device = x.device

        # Masque causal L x L
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

    Architecture :
    Input (tokens ids) →
        PositionalEmbedding →
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

        self.embedding = PositionalEmbedding(vocab_size, embed_dim, max_length)
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
        # Embedding (batch, seq_len, embed_dim)
        x = self.embedding(x)

        # Empilement des blocs Transformer
        for block in self.blocks:
            x = block(x)

        # Dropout global
        x = self.dropout(x)

        # On ne garde que la dernière position
        last_token = x[:, -1, :]  # (batch, embed_dim)

        # Logits vocabulaire
        logits = self.fc_out(last_token)  # (batch, vocab_size)
        return logits


# ======================================================================
#  GÉNÉRATEUR DE TEXTE : TransformerTextGenerator
# ======================================================================


class TransformerTextGenerator:
    """
    Générateur de texte pour le modèle Transformer (PyTorch, checkpoint v2).

    - Charge le corpus texte depuis PROJECT_ROOT / data/raw/nlp
    - Nettoie le texte comme dans le notebook
    - Reconstruit le tokenizer (word-level) identique
    - Reconstruit le modèle PyTorch si on reçoit un state_dict (OrderedDict)
    - Génère du texte en mode auto-régressif à partir d'un seed.
    """

    # Singleton interne
    _instance: "TransformerTextGenerator | None" = None

    # Références au projet / corpus
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CORPUS_DIR = PROJECT_ROOT / "gradio" / "data"
    CORPUS_LENGTH_PARAM = "_car_FULL_"  # même filtre que dans le notebook FULL_50

    # Longueur de séquence par défaut (notebook FULL_50)
    DEFAULT_MAX_LENGTH = 50

    def __init__(self, model, tokenizer: Tokenizer, max_length: int):
        # Ici, on veut un nn.Module, pas un state_dict
        if isinstance(model, (dict, OrderedDict)):
            raise TypeError(
                "TransformerTextGenerator a reçu un 'state_dict' (OrderedDict) au lieu d'un modèle. "
                "Reconstruction du modèle attendue AVANT l'instanciation."
            )

        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Backend: PyTorch ou Keras (théorique, mais ici on est en PyTorch)
        self._is_torch = isinstance(self.model, nn.Module)
        if self._is_torch:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
        else:
            self.device = None

    # ------------------------------------------------------------------
    # Méthodes de classe : singleton / factory
    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls, model_or_state) -> "TransformerTextGenerator":
        """
        Retourne une unique instance, basée sur :
        - un modèle Transformer PyTorch déjà construit (nn.Module)
        - OU un state_dict (OrderedDict) provenant du .pt sauvegardé.
        """

        if cls._instance is not None:
            return cls._instance

        # 1) Reconstruire le modèle PyTorch si on reçoit un state_dict
        if isinstance(model_or_state, (dict, OrderedDict)):
            torch_model = cls._build_torch_model_from_state_dict(model_or_state)
        else:
            # On suppose ici un nn.Module déjà prêt (PyTorch)
            torch_model = model_or_state

        # 2) Charger + nettoyer le corpus, rebuild du tokenizer
        full_text_clean = cls._load_full_clean_corpus()
        tokenizer = cls._build_tokenizer_from_corpus(full_text_clean)

        # 3) max_length : si le modèle l'expose, on le récupère, sinon fallback
        max_length = getattr(torch_model, "max_length", cls.DEFAULT_MAX_LENGTH)

        cls._instance = cls(
            model=torch_model,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        return cls._instance

    # ------------------------------------------------------------------
    # Reconstruction du modèle PyTorch depuis le state_dict
    # ------------------------------------------------------------------
    @classmethod
    def _build_torch_model_from_state_dict(
        cls,
        state: Mapping[str, torch.Tensor],
    ) -> TransformerLanguageModel:
        """
        Reconstruit un `TransformerLanguageModel` à partir d’un state_dict
        tel que sauvegardé dans le notebook transformer_trainme_v2.

        On infère les hyperparamètres structurants à partir des shapes :
        - vocab_size        : dim 0 de embedding.token_emb.weight
        - embed_dim         : dim 1 de embedding.token_emb.weight
        - ff_dim            : dim 0 de blocks.0.ffn.0.weight
        - num_blocks        : nombre de blocs présents dans `blocks.{i}.att.query.weight`
        - num_heads         : on reprend la valeur du notebook (4)
        - max_length        : on utilise DEFAULT_MAX_LENGTH (50) utilisée lors du training.
        """

        # Vérifications de base
        if "embedding.token_emb.weight" not in state:
            raise KeyError(
                "Le state_dict fourni ne contient pas la clé 'embedding.token_emb.weight'. "
                "Vérifie que tu utilises bien le checkpoint du TransformerLanguageModel v2."
            )

        # vocab_size, embed_dim
        token_emb_weight = state["embedding.token_emb.weight"]
        vocab_size = token_emb_weight.shape[0]
        embed_dim = token_emb_weight.shape[1]

        # ff_dim
        ffn0_key = "blocks.0.ffn.0.weight"
        if ffn0_key not in state:
            raise KeyError(
                f"Clé '{ffn0_key}' absente du state_dict. "
                "La structure attendue est blocks.{i}.ffn.0.weight."
            )
        ff_dim = state[ffn0_key].shape[0]

        # Nombre de blocs
        num_blocks = 0
        while f"blocks.{num_blocks}.att.query.weight" in state:
            num_blocks += 1
        if num_blocks == 0:
            raise ValueError(
                "Impossible de déterminer le nombre de blocs Transformer "
                "(aucune clé 'blocks.{i}.att.query.weight' trouvée)."
            )

        # Dans le notebook v2, num_heads = 4
        num_heads = 4

        # max_length : utilisé pour le positional encoding, non paramétrique
        max_length = cls.DEFAULT_MAX_LENGTH

        # Dropout rate standard du notebook
        dropout_rate = 0.1

        # Construction du modèle
        model = TransformerLanguageModel(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_blocks=num_blocks,
            dropout_rate=dropout_rate,
        )

        # Chargement des poids
        model.load_state_dict(state)
        return model

    # ------------------------------------------------------------------
    # Chargement du corpus & nettoyage
    # ------------------------------------------------------------------
    @classmethod
    def _load_full_clean_corpus(cls) -> str:
        """
        Reproduit la logique du notebook :

        - charge les .txt dans CORPUS_DIR
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
    # Backend-unified prediction
    # ------------------------------------------------------------------
    def _predict_proba(self, sequence: np.ndarray) -> np.ndarray:
        """
        Unifie la prédiction entre Keras (.predict) et PyTorch (.forward).
        Retourne un vecteur de probabilités sur le vocabulaire.
        """
        if self._is_torch:
            x = torch.from_numpy(sequence).long().to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                # Ici, logit final : (batch, vocab_size)
                if logits.dim() == 2:
                    logits = logits[0]  # (vocab_size,)
                else:
                    # fallback très défensif
                    logits = logits.view(-1)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            return probs
        else:
            # Chemin Keras théorique (non utilisé dans v2)
            return self.model.predict(sequence, verbose=0)[0]

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
        """
        Génère du texte à partir d’un prompt initial, en mode auto-régressif.

        - temperature contrôle la créativité
        - num_words = nombre de tokens supplémentaires à générer
        """
        seed_text = seed_text.strip()
        if not seed_text:
            return ""

        rng = np.random.default_rng(seed) if seed is not None else np.random

        generated_text = seed_text
        token_list = self._encode_prompt(seed_text)

        for _ in range(num_words):
            if not token_list:
                break

            # Padding / tronquage à max_length-1 comme au training
            sequence = pad_sequences(
                [token_list],
                maxlen=self.max_length - 1,
                padding="pre",
            )

            # Prédiction du prochain token (probas)
            preds = self._predict_proba(sequence.astype("int64"))

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
