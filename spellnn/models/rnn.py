from typing import Tuple

from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dense, Embedding, TimeDistributed


class RNNSpellChecker(Model):
    def __init__(self,
                 input_shape: Tuple = (None,),
                 nb_classes: int = None,
                 embeddings_size: int = 8,
                 dropout: float = 0.2,
                 hidden_units: Tuple[int, ...] = (512, 512, 512),
                 inputs=None, outputs=None, name='SpellChecker'):
        if inputs or outputs:
            super().__init__(inputs=inputs, outputs=outputs, name=name)
            return

        net_input = Input(shape=input_shape, dtype='int64', name='input')
        x = Embedding(input_dim=nb_classes, output_dim=embeddings_size, name='char_embeddings')(net_input)
        for units in hidden_units:
            x = LSTM(units=units, recurrent_dropout=dropout, return_sequences=True)(x)

        net_output = TimeDistributed(Dense(nb_classes, activation='softmax', name='output'))(x)
        super().__init__(inputs=net_input, outputs=net_output, name=name)
