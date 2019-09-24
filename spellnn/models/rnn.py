from typing import Tuple, Collection

import tensorflow as tf
from spellnn.layers.mapping import CharMapping

keras = tf.keras
from keras import Input, Model
from keras.layers import LSTM, Dense, Embedding


class RNNSpellChecker(Model):
    def __init__(self,
                 chars: Collection[str] = None,
                 embeddings_size: int = 8,
                 dropout: float = 0.2,
                 hidden_units: Tuple[int, ...] = (512, 512, 512),
                 inputs=None, outputs=None, name='SpellChecker'):
        if inputs or outputs:
            super().__init__(inputs=inputs, outputs=outputs, name=name)
            return

        nb_classes = len(chars) + 1
        net_input = Input(shape=(None,), dtype='string', name='input')
        x = CharMapping(chars=chars, include_unknown=True, name='char_mapping')(net_input)
        x = Embedding(input_dim=nb_classes, output_dim=embeddings_size, name='char_embeddings')(x)
        for units in hidden_units:
            x = LSTM(units=units, recurrent_dropout=dropout, return_sequences=True)(x)

        net_output = Dense(units=nb_classes, activation='softmax', name='output')(x)
        super().__init__(inputs=net_input, outputs=net_output, name=name)
