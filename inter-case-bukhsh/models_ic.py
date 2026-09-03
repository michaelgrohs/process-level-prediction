# -*- coding: utf-8 -*-
"""
Remaining-time transformer model with a variable-width numeric feature
input.

processtransformer/models/transformer.py's own get_remaining_time_model()
hardcodes ``time_inputs = layers.Input(shape=(5,))`` (the 5 base timing
features). Everything else there -- token/position embedding, the
transformer block, layer sizes -- is reused UNMODIFIED (imported directly);
this file only swaps in a variable-width time-feature input so the model
can also take the extra inter-case ("load state") columns from
features.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

PT_DIR = Path(__file__).resolve().parent.parent / "processtransformer"
if str(PT_DIR) not in sys.path:
    sys.path.insert(0, str(PT_DIR))

import tensorflow as tf  # noqa: E402
from tensorflow.keras import layers  # noqa: E402

from models.transformer import TokenAndPositionEmbedding, TransformerBlock  # noqa: E402


def get_remaining_time_model_ic(
    max_case_length: int,
    vocab_size: int,
    num_time_feats: int,
    embed_dim: int = 36,
    num_heads: int = 4,
    ff_dim: int = 64,
) -> tf.keras.Model:
    inputs = layers.Input(shape=(max_case_length,))
    time_inputs = layers.Input(shape=(num_time_feats,))

    x = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)(inputs)
    x = TransformerBlock(embed_dim, num_heads, ff_dim)(x, training=True)
    x = layers.GlobalAveragePooling1D()(x)

    x_t = layers.Dense(32, activation="relu")(time_inputs)
    x = layers.Concatenate()([x, x_t])
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="linear")(x)

    return tf.keras.Model(
        inputs=[inputs, time_inputs], outputs=outputs,
        name="remaining_time_transformer_ic",
    )
