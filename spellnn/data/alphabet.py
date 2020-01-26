import string
import homoglyphs as hg


START = '<S>'
END = '</E>'


def get_chars(locale):
    return list(dict.fromkeys(
        list(string.printable) +
        list(hg.Languages.get_alphabet([locale]))
    ))
