from typing import Dict, Iterable

import tensorflow as tf

from ..data import Field
from . import utils
from ..layers import CNNEncoder


def CNNModel(features: Dict[str, Field],
             targets: Dict[str, Fielf],
             text_embedder,
             filters: int = 150,
             kernel_sizes: Iterable[int] = (2, 3, 4, 5),
             conv_layer_activation='relu',
             l1_regularization: float = None,
             l2_regularization: float = None,
             dropout: float = 0.5,
             activation='softmax'):
    inputs = utils.create_inputs(features)
    embedded_token = text_embedder(inputs['token'])

    pos_head_embedding = tf.keras.layers.Embedding(len(features['pos_head'].vocab),
                                                   50, mask_zero=True)
    pos_tail_embedding = tf.keras.layers.Embedding(len(features['pos_tail'].vocab),
                                                   50, mask_zero=True)
    embedded_pos_head = pos_head_embedding(inputs['pos_head'])
    embedded_pos_tail = pos_tail_embedding(inputs['pos_tail'])
    x = tf.keras.layers.Concatenate()([embedded_token, embedded_pos_head, embedded_pos_tail])

    cnn_encoder = CNNEncoder(filters, kernel_sizes,
                             conv_layer_activation,
                             l1_regularization,
                             l2_regularization)
    x = cnn_encoder(x)
    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)
    probs = tf.keras.layers.Dense(
        len(targets['relation'].vocab), activation=activation, name='relation')(x)
    return tf.keras.models.Model(inputs=list(inputs.values()),
                                 outputs=probs,
                                 name="CNNModel")
