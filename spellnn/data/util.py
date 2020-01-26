from itertools import takewhile, repeat
from pathlib import Path
from typing import Union


def nb_lines(file_path: Union[str, Path]):
    with open(file_path, 'rb') as f:
        bufgen = takewhile(lambda x: x, (f.raw.read(1024 * 1024) for _ in repeat(None)))
        return sum(buf.count(b'\n') for buf in bufgen)
