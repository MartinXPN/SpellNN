import string
import homoglyphs as hg


def get_chars(locale):
    return list(dict.fromkeys(
        list(string.printable) +
        list(hg.Languages.get_alphabet([locale]))
    ))
