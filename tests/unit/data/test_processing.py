import string
import unittest
from typing import List

import numpy as np

from spellnn.data import alphabet
from spellnn.data.processing import DataProcessor


class DataProcessingTest(unittest.TestCase):

    def setUp(self):
        self.processor = DataProcessor(locale='en', char2id=self.map_to_id, alphabet=string.printable)
        self.char2id = {c: i for i, c in enumerate([alphabet.START, alphabet.END] + list(string.printable))}

    def map_to_id(self, text: List[str]):
        res = [self.char2id.get(c, 0) for c in text]
        return np.array(res)

    def test_one_line_processing(self):
        line = "Hi,my name is Slim     Shady.How're you?"
        encoder_input, decoder_input, target = self.processor.to_sample(line)
        print(f'{line} -> {encoder_input} ====  {decoder_input} ====  {target}')
        print(f"{line} -> {''.join(encoder_input)} ==== {''.join(decoder_input)} ====  {''.join(target)}")
        self.assertListEqual(decoder_input + [alphabet.END], [alphabet.START] + target)
        self.assertTrue(any(a != b for a, b in zip(list(line), encoder_input)))

    def test_batch(self):
        batch = np.array([b'Hi, this is the first sentence',
                          b'This is the second one',
                          b'We can have several more, but we will stop here!'])
        res = self.processor.process_batch(batch)
        print([r.shape for r in res])
        self.assertEqual(len(res), 3)   # encoder_input, decoder_input, target
        for x in res:
            self.assertEqual(x.shape[0], batch.shape[0])

        encoder_input, decoder_input, target = res
        print('Encoder:', encoder_input)
        print('Decoder:', decoder_input)
        print('Target:', target)

        np.testing.assert_equal(decoder_input[:, 0], np.ones(3) * self.char2id[alphabet.START])
        np.testing.assert_equal(target[:, -1], np.ones(3) * self.char2id[alphabet.END])

    def test_doc_reset(self):
        self.processor.cache_limit = 2
        self.test_batch()
        print('CACHE LIMIT:', self.processor.cache_limit)
        print('BATCH DOCS:', len(self.processor.batch_docs))
        self.test_batch()
        print(self.processor.batch_docs)
        self.assertLessEqual(len(self.processor.batch_docs), 3)  # Batch size
