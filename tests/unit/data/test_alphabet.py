import unittest

from spellnn.data.alphabet import get_chars


class AlphabetTest(unittest.TestCase):

    def test_en_alphabet(self):
        chars = get_chars(locale='en')
        print(f'locale: en -> {chars}')
        self.assertEqual(len(chars), len(set(chars)), 'Alphabet has to have only unique characters')
