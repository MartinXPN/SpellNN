from typing import Iterable, Optional

import tensorflow as tf
from tensorflow.keras.layers import Layer


class CharMapping(Layer):
    UNK = '<UNK>'

    def __init__(self,
                 chars: Iterable[str],
                 include_unknown: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.chars = []
        if include_unknown:
            self.chars += [self.UNK]
        self.chars += chars
        self.chars = list(dict.fromkeys(self.chars))

        self.table = tf.lookup.StaticHashTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=tf.constant(self.chars),
                values=tf.constant(list(range(len(self.chars)))),
            ),
            default_value=0,
        )

    def call(self, inputs, **kwargs):
        return self.table.lookup(inputs)

    def get_config(self):
        config = {'chars': self.chars}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
