import string
import unittest

from nltk import edit_distance

from spellnn.data.spelling_mistakes import Mistakes


class SpellingMistakesTest(unittest.TestCase):
    def setUp(self):
        self.alphabet = string.printable
        self.mistakes = Mistakes(alphabet=self.alphabet)
        print(f'alphabet: {self.alphabet}')

    def test_delete(self):
        s = 'abcdefgh'

        position = 3
        res = self.mistakes.delete(s, start=position, end=position)
        print(f'delete: `{s}` -> `{res}`')
        self.assertEqual(res, s[:position] + s[position + 1:])
        self.assertEqual(edit_distance(s, res), 1)

        res = self.mistakes.delete(s)
        print(f'delete: `{s}` -> `{res}`')
        self.assertEqual(edit_distance(s, res), 1)
        self.assertEqual(len(s) - 1, len(res))
        self.assertEqual(edit_distance(s, res), 1)

    def test_insert(self):
        s = 'efghklmn'
        res = self.mistakes.insert(s)
        print(f'insert: `{s}` -> `{res}`')
        self.assertEqual(len(s) + 1, len(res))
        self.assertEqual(edit_distance(s, res), 1)

    def test_swap(self):
        s = 'swapping stuff!'
        res = self.mistakes.swap(s)
        print(f'swap: `{s}` -> `{res}`')
        self.assertEqual(len(s), len(res))
        self.assertLessEqual(edit_distance(s, res), 2)

    def test_replace(self):
        s = 'replace a character?'
        res = self.mistakes.replace(s)
        print(f'swap: `{s}` -> `{res}`')
        self.assertEqual(len(s), len(res))
        self.assertLessEqual(edit_distance(s, res), 1)
