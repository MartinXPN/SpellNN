import math
import random
from typing import List


class Mistakes:
    def __init__(self, alphabet: List[str]):
        self.alphabet = alphabet

    def delete(self, text: str, start: int = 0, end: int = -1):
        if text == '':
            return ''
        if end < 0:
            end += len(text)
        i = random.randint(start, end)
        return text[:i] + text[i + 1:]

    def swap(self, text: str, start: int = 0, end: int = -1):
        if len(text) < 2:
            return text
        if end < 0:
            end += len(text)
        chars = list(text)

        end = min(end, len(text) - 2)
        i = random.randint(start, end)
        j = i + 1

        chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)

    def insert(self, text: str, start: int = 0, end: int = -1):
        if end < 0:
            end += len(text)

        c = random.choice(self.alphabet)
        i = random.randint(start, end)
        return text[:i] + c + text[i:]

    def replace(self, text: str, start: int = 0, end: int = -1):
        if text == '':
            return ''
        if end < 0:
            end += len(text)

        c = random.choice(self.alphabet)
        i = random.randint(start, end)
        return text[:i] + c + text[i + 1:]


def apply_spelling_errors(text: str, mistakes: Mistakes):
    """
    Apply random spelling errors to the given text.
    For now, the implementation uses simple heuristic to determine which errors to pick.
    number of errors is at most log2( len(text) ).

    In future this has to:
        * Not modify named entities
        * Stage 1 - generate possible modifications that could be applied to the text (word or span)
        * Stage 2 - pick modifications by probability and apply randomly
    """
    nb_errors = int(math.log2(len(text)))
    modifications = random.choices(population=[mistakes.delete, mistakes.insert, mistakes.replace, mistakes.swap],
                                   weights=[0.2, 0.3, 0.3, 0.4],
                                   k=nb_errors)
    for modification in modifications:
        text = modification(text)
    return text
