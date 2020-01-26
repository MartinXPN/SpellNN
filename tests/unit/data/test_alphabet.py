import unittest

from spellnn.data.alphabet import get_chars, START, END


class AlphabetTest(unittest.TestCase):

    def test_en_alphabet(self):
        chars = get_chars(locale='en')
        print(f'locale: en -> {chars}')
        self.assertEqual(len(chars), len(set(chars)), 'Alphabet has to have only unique characters')

    def test_start_end(self):
        self.assertNotEqual(START, END)
