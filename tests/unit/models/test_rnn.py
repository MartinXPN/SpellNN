from pathlib import Path
from unittest import TestCase

import numpy as np

from spellnn.models.rnn import RNNSpellChecker

import tensorflow as tf

keras = tf.keras
from keras.utils import CustomObjectScope
from keras.models import load_model


class TestRNNModel(TestCase):

    def test_rnn_construction(self):
        model = RNNSpellChecker()
        model.summary()

    def test_input_output_shape(self):
        model = RNNSpellChecker()
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        model.summary()

        inputs = np.ones(shape=(1, 9), dtype='uint8')
        outputs = model.predict(inputs)
        self.assertEqual(outputs.shape, (1, 9, 64))


class TestLoadSave(TestCase):
    def setUp(self):
        self.model = RNNSpellChecker()
        self.model.compile(optimizer='adam', loss='categorical_crossentropy')
        self.file_path = Path('rnn-model.h5')
        self.tf_lite_path = Path('rnn-model.tflite')

    def test_save_load(self):
        inputs = np.ones(shape=(1, 9), dtype='uint8')
        before_saving = self.model.predict(inputs)

        self.model.save(filepath=self.file_path)
        with CustomObjectScope({'RNNSpellChecker': RNNSpellChecker}):
            model = load_model(self.file_path)
            self.assertIsInstance(model, RNNSpellChecker)

        after_saving = model.predict(inputs)
        self.assertTrue(np.array_equal(before_saving, after_saving))

    def test_tf_lite(self):
        self.skipTest('Current version of tf and keras don\'t support exporting to tf.lite')
        self.model.save(self.file_path)
        with CustomObjectScope({'RNNSpellChecker': RNNSpellChecker}):
            model = load_model(self.file_path)

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
