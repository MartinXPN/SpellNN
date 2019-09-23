import unittest

import numpy as np
import tensorflow as tf

from spellnn.layers.mapping import CharMapping

keras = tf.keras
from keras import Input
from keras import Model
from keras.utils.test_utils import layer_test
from keras.utils import CustomObjectScope


class MappingLayerTest(unittest.TestCase):
    def test_save_load_execution(self):
        batch_size = 2
        nb_chars = 12

        with CustomObjectScope({'CharMapping': CharMapping}):
            output = layer_test(CharMapping,
                                kwargs={'chars': ['<s>', 'b', 'i', 'k', 'c', 'т']},
                                input_shape=(batch_size, nb_chars),
                                input_dtype=np.string_,
                                expected_output_dtype=tf.int32)
            self.assertEqual(np.array(output).shape, (batch_size, nb_chars))

            output = layer_test(CharMapping,
                                kwargs={'chars': ['<s>', 'b', 'i', 'k', 'c', 'т'], 'include_unknown': False},
                                input_shape=(batch_size, 5, nb_chars),
                                input_dtype=np.string_,
                                expected_output_dtype=tf.int32)
            self.assertEqual(np.array(output).shape, (batch_size, nb_chars))

    def test_mapping(self):
        chars = ['<s>', 'b', 'i', 'k', 'c', 'т']
        i = Input(shape=(None,), dtype='string', name='input')
        x = CharMapping(chars=chars, include_unknown=False)(i)
        model = Model(inputs=i, outputs=x)

        inputs = [chars[2], chars[4], chars[0], 'vpu']
        mapped = model.predict(np.array([inputs]))[0]

        print('Inputs:', inputs)
        print('Results:', mapped)
        self.assertEqual(mapped[0], 2)
        self.assertEqual(mapped[1], 4)
        self.assertEqual(mapped[2], 0)
        self.assertEqual(mapped[3], 0)

        self.assertEqual(model.predict(np.array(['c']))[0][0], chars.index('c'))
        self.assertEqual(model.predict(np.array(['т']))[0][0], chars.index('т'))
        self.assertEqual(model.predict(np.array(['<s>']))[0][0], chars.index('<s>'))
