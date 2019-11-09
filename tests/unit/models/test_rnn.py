from pathlib import Path
from unittest import TestCase

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from spellnn.models.rnn import RNNSpellChecker


class TestRNNModel(TestCase):

    def setUp(self):
        self.chars = ['<s>', 'b', 'v', 'c', 'т', '<UNK>']
        self.nb_classes = len(self.chars)
        self.model = RNNSpellChecker(nb_classes=self.nb_classes)

    def test_rnn_construction(self):
        self.model.summary()

    def test_input_output_shape(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.model.summary()

        inputs = np.ones(shape=(1, 15))
        outputs = self.model.predict(inputs)
        print(outputs)
        self.assertEqual(outputs.shape, inputs.shape + (self.nb_classes,))


class TestLoadSave(TestCase):
    def setUp(self):
        self.chars = ['<s>', 'b', 'v', 'c', 'т', '<UNK>']
        self.nb_classes = len(self.chars)
        self.model = RNNSpellChecker(nb_classes=self.nb_classes)
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.file_path = Path('rnn-model.h5')
        self.tf_lite_path = Path('rnn-model.tflite')

    def test_save_load(self):
        inputs = np.random.rand(1, 15)
        before_saving = self.model.predict(inputs)

        self.model.save(filepath=self.file_path)
        with CustomObjectScope({'RNNSpellChecker': RNNSpellChecker}):
            model = load_model(self.file_path)
            self.assertIsInstance(model, RNNSpellChecker)

        after_saving = model.predict(inputs)
        self.assertTrue(np.array_equal(before_saving, after_saving))

    def test_tf_lite(self):
        self.skipTest('The current version of tf and keras don\'t support exporting LSTM layers to tf.lite')
        self.model.save(self.file_path)
        with CustomObjectScope({'RNNSpellChecker': RNNSpellChecker}):
            model = load_model(self.file_path)
            # TF-lite demands to have a fixed size inputs for all the inputs other than the batch dimension
            fixed_input = Input(shape=(512,))
            fixed_output = model(fixed_input)
            model = Model(fixed_input, fixed_output)

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
