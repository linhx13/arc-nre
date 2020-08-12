from typing import Dict

import tensorflow as tf

from ..data import Field


def create_inputs(features: Dict[str, Field]):
    inputs = {n: tf.keras.layers.Input(shape=(f.max_len,), name=n)
              for n, f in features.items()}
    return inputs
