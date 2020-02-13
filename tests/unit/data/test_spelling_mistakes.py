import string
import unittest
from textwrap import dedent

from nltk import edit_distance

from spellnn.data.spelling_mistakes import Mistakes, apply_spelling_errors


class SpellingMistakesTest(unittest.TestCase):
    def setUp(self):
        self.alphabet = list(string.printable)
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

        self.assertEqual(self.mistakes.delete(''), '')

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
        self.assertEqual(self.mistakes.swap(''), '')
        self.assertEqual(self.mistakes.swap('a'), 'a')

    def test_replace(self):
        s = 'replace a character?'
        res = self.mistakes.replace(s)
        print(f'swap: `{s}` -> `{res}`')
        self.assertEqual(len(s), len(res))
        self.assertLessEqual(edit_distance(s, res), 1)
        self.assertEqual(self.mistakes.replace(''), '')


class TextSpellingMistakesApplicationTest(unittest.TestCase):

    def setUp(self):
        self.alphabet = list(string.printable)
        self.mistakes = Mistakes(alphabet=self.alphabet)

    def test_random_string(self):
        s = 'some pretty long string that will be modified'
        res = apply_spelling_errors(s, mistakes=self.mistakes)
        self.assertNotEqual(s, res)

    def test_long_string(self):
        s = dedent('''
        Here I am, alone again
        Can't get out of this hole I'm in
        It's like the walls are closin' in
        You can't help me, no one can
        I can feel these curtains closin'
        I go to open 'em
        But something pulls 'em closed again
        (Hello darkness, my old friend)
        Feels like I'm loathing in Las Vegas
        Haven't got the vaguest why I'm so lost
        But I'd make you this small wager
        If I bet you, I'll be in tomorrow's paper
        Who would the odds favor?
        (Hello darkness, my old friend)
        I'm so much like my father, you would think that I knew him
        I keep pacin' this room valium
        Then chase it with booze, one little taste it'll do
        Maybe I'll take it and snooze, then tear up the stage in a few
        Fuck the Colt 45, I'ma need somethin' stronger
        If I pop any caps, it better be off of vodka
        Round after round after round, I'm gettin' loaded
        That's a lot of shots, huh? (double entendre)
        (Hello darkness, my old friend)
        
        And I don't wanna be alone in the darkness
        I don't wanna be alone in the darkness
        I don't wanna be alone in the darkness anymore
        (Hello darkness, my old friend)
        ''')
        res = apply_spelling_errors(s, mistakes=self.mistakes)
        print(res)
        self.assertNotEqual(s, res)
