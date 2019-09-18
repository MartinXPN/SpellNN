from unittest import TestCase

import numpy as np

from spellnn.models.rnn import RNNSpellChecker


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
