from typing import Tuple

import tensorflow as tf
keras = tf.keras
from keras import Input, Model
from keras.layers import LSTM, Dense, Embedding


class RNNSpellChecker(Model):
    def __init__(self,
                 nb_input_chars: int = 64, nb_output_chars: int = 64,
                 embeddings_size: int = 8,
                 dropout: float = 0.2,
                 hidden_units: Tuple[int, ...] = (512, 512, 512),
                 inputs=None, outputs=None, name='SpellChecker'):
        if inputs or outputs:
            super().__init__(inputs=inputs, outputs=outputs, name=name)
            return

        net_input = Input(shape=(None,), dtype='uint8', name='input')
        x = Embedding(input_dim=nb_input_chars, output_dim=embeddings_size, name='char_embeddings')(net_input)
        for units in hidden_units:
            x = LSTM(units=units, recurrent_dropout=dropout, return_sequences=True)(x)

        net_output = Dense(units=nb_output_chars, activation='softmax', name='output')(x)
        super().__init__(inputs=net_input, outputs=net_output, name=name)
