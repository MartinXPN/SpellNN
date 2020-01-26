from pathlib import Path
from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from spellnn.models import Seq2SeqAttentionCNN


class TestSeq2SeqAttentionCNN(TestCase):

    def setUp(self):
        self.chars = ['<s>', 'b', 'v', 'c', 'т', '<UNK>']
        self.model = Seq2SeqAttentionCNN(nb_symbols=len(self.chars))

    def test_model_construction(self):
        self.model.summary()

    def test_input_output_shape(self):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        self.model.summary()

        encoder_inputs = np.ones(shape=(1, 15))
        decoder_inputs = np.ones(shape=(1, 17))

        outputs = self.model.predict([encoder_inputs, decoder_inputs])
        print(outputs)
        self.assertEqual(outputs.shape, decoder_inputs.shape + (len(self.chars),))


class TestLoadSave(TestCase):
    def setUp(self):
        self.chars = ['<s>', 'b', 'v', 'c', 'т', '<UNK>']
        self.nb_classes = len(self.chars)
        self.model = Seq2SeqAttentionCNN(nb_symbols=len(self.chars))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.file_path = Path('attn-cnn-model.h5')
        self.tf_lite_path = Path('attn-cnn-model.tflite')

    def test_save_load(self):
        encoder_inputs = np.ones(shape=(1, 15))
        decoder_inputs = np.ones(shape=(1, 17))

        before_saving = self.model.predict([encoder_inputs, decoder_inputs])

        self.model.save(filepath=self.file_path)
        with CustomObjectScope({'Seq2SeqAttentionCNN': Seq2SeqAttentionCNN}):
            model = load_model(self.file_path)
            self.assertIsInstance(model, Seq2SeqAttentionCNN)

        after_saving = self.model.predict([encoder_inputs, decoder_inputs])
        self.assertTrue(np.array_equal(before_saving, after_saving))

    def test_tf_lite(self):
        self.skipTest('Here is a list of operators for which you will need custom implementations: BatchMatMul.')
        self.model.save(self.file_path)
        with CustomObjectScope({'Seq2SeqAttentionCNN': Seq2SeqAttentionCNN}):
            model = load_model(self.file_path)
            # TF-lite demands to have a fixed size inputs for all the inputs other than the batch dimension
            fixed_encoder_input = Input(shape=(512,))
            fixed_decoder_input = Input(shape=(512,))
            fixed_output = model([fixed_encoder_input, fixed_decoder_input])
            model = Model([fixed_encoder_input, fixed_decoder_input], fixed_output)

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tf_lite_model = converter.convert()
        with open(self.tf_lite_path, 'wb') as f:
            f.write(tf_lite_model)

        print(tf_lite_model)
        print(self.tf_lite_path.stat())
        self.assertLessEqual(self.tf_lite_path.stat().st_size, 10000000)

    def doCleanups(self):
        if self.file_path.exists():
            self.file_path.unlink()
        if self.tf_lite_path.exists():
            self.tf_lite_path.unlink()
