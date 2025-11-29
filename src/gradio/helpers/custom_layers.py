import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Pour la sérialisation propre des couches custom
try:
    from keras.saving import register_keras_serializable
except ImportError:
    from tensorflow.keras.utils import register_keras_serializable


def get_positional_encoding(seq_len, d_model):
    """
    Crée le positional encoding pour le Transformer (sinus/cosinus).
    """
    positions = np.arange(seq_len)[:, np.newaxis]
    dimensions = np.arange(d_model)[np.newaxis, :]
    angle_rates = 1 / np.power(10000, (2 * (dimensions // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates

    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(angle_rads[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(pos_encoding[np.newaxis, ...], dtype=tf.float32)


@register_keras_serializable(package="trainme")
class MultiHeadSelfAttention(layers.Layer):
    """
    Implémentation du Multi-Head Self-Attention (causal).
    Même logique que dans le notebook d'entraînement.
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        assert embed_dim % num_heads == 0, "embed_dim doit être divisible par num_heads"

        self.projection_dim = embed_dim // num_heads

        # Projections linéaires pour Q, K, V
        self.query_dense = layers.Dense(embed_dim, name="query")
        self.key_dense = layers.Dense(embed_dim, name="key")
        self.value_dense = layers.Dense(embed_dim, name="value")

        # Projection finale
        self.combine_heads = layers.Dense(embed_dim, name="output")

    def attention(self, query, key, value):
        """Calcule l'attention scaled dot-product avec masque de causalité."""
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)

        # Masque de causalité (look-ahead mask)
        seq_len = tf.shape(scaled_score)[-1]
        mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = mask * -1e9  # -inf sur les positions futures

        scaled_score += mask

        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """Sépare la dernière dimension en (num_heads, projection_dim)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]

        # Projections linéaires
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)

        # Séparer en multiple heads
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        # Attention avec masque de causalité
        attention, weights = self.attention(query, key, value)

        # Recombiner les heads
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))

        # Projection finale
        output = self.combine_heads(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
            }
        )
        return config


@register_keras_serializable(package="trainme")
class TransformerBlock(layers.Layer):
    """
    Un bloc Transformer complet, identique au notebook :
    - Multi-Head Self-Attention (custom)
    - Feed-Forward Network
    - Layer Normalization
    - Residual Connections
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Multi-Head Attention (ta classe custom)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        # Feed-Forward Network
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

        # Layer Normalization
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-Head Attention avec residual connection
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-Forward Network avec residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


@register_keras_serializable(package="trainme")
class PositionalEmbedding(layers.Layer):
    """
    Combine l'embedding des tokens avec le positional encoding.
    Identique au notebook, avec compat sérialisation.
    """
    def __init__(self, vocab_size, embed_dim, max_len, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_len = max_len

        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_encoding = get_positional_encoding(max_len, embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        x = self.token_emb(x)
        # Ajouter le positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        return x

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
