from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Embedding, Conv1D, Dot, Activation, Concatenate


class Seq2SeqAttentionCNN(Model):
    def __init__(self,
                 nb_symbols: int = 256,
                 embedding_size: int = 8,
                 inputs=None, outputs=None, name='SpellChecker'):
        if inputs or outputs:
            super().__init__(inputs=inputs, outputs=outputs, name=name)
            return

        # Encoder
        encoder_inputs = Input(shape=(None,), dtype='uint8', name='encoder_input')
        enc_char_embeddings = Embedding(nb_symbols, embedding_size, name='encoder_char_embeddings')(encoder_inputs)
        x_encoder = Conv1D(256, kernel_size=3, activation='relu', padding='causal')(enc_char_embeddings)
        x_encoder = Conv1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=2)(x_encoder)
        x_encoder = Conv1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=4)(x_encoder)

        # Decoder
        decoder_inputs = Input(shape=(None,), dtype='uint8', name='decoder_input')
        dec_char_embeddings = Embedding(nb_symbols, embedding_size, name='decoder_char_embeddings')(decoder_inputs)
        x_decoder = Conv1D(256, kernel_size=3, activation='relu', padding='causal')(dec_char_embeddings)
        x_decoder = Conv1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=2)(x_decoder)
        x_decoder = Conv1D(256, kernel_size=3, activation='relu', padding='causal', dilation_rate=4)(x_decoder)

        # Attention
        attention = Dot(axes=[2, 2])([x_decoder, x_encoder])
        attention = Activation('softmax')(attention)

        context = Dot(axes=[2, 1])([attention, x_encoder])
        decoder_combined_context = Concatenate(axis=-1)([context, x_decoder])

        decoder_outputs = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(decoder_combined_context)
        decoder_outputs = Conv1D(64, kernel_size=3, activation='relu', padding='causal')(decoder_outputs)
        decoder_outputs = Dense(nb_symbols, activation='softmax')(decoder_outputs)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()

        super().__init__(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs, name=name)
