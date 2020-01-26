import string
from typing import List

import homoglyphs as hg


START = '<S>'
END = '</E>'


def get_chars(locale: str) -> List[str]:
    return list(dict.fromkeys(
        list(string.printable) +
        list(hg.Languages.get_alphabet([locale]))
    ))
