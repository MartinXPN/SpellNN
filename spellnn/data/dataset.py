import abc
import atexit
import random
from functools import partial
from pathlib import Path
from typing import List, Iterator, overload, Union

from spellnn.entities import Sample


class Dataset:
    """
    Interface for a dataset.
    Any subclass implementing the abstract methods of this class will work within SpellNN
    """
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        :return: number of samples in the dataset
        """

    @abc.abstractmethod
    def __iter__(self) -> Iterator[Sample]:
        """
        :return: iter(self.data)
        """

    @abc.abstractmethod
    def shuffle(self):
        """
        Shuffle the dataset (usually called in between the epochs)
        """

    @overload
    def __getitem__(self, i: slice) -> List[Sample]:
        ...

    @overload
    def __getitem__(self, i: int) -> Sample:
        ...

    @abc.abstractmethod
    def __getitem__(self, i):
        pass


class PlainTextFileDataset(Dataset):

    def __init__(self, file_path: Union[str, Path]):
        self.data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = [Sample(text=line.strip()) for line in f]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self) -> Iterator[Sample]:
        return iter(self.data)

    def shuffle(self):
        random.shuffle(self.data)

    @overload
    def __getitem__(self, i: slice) -> List[Sample]:
        ...

    @overload
    def __getitem__(self, i: int) -> Sample:
        ...

    def __getitem__(self, i):
        return self.data[i]


class LargeTextFileDataset(Dataset, abc.ABC):
    # TODO
    """
    As plain text files can be large and reach up to several GB in storage,
    this class implements lazy loading of the dataset file.
    """
    def __init__(self, file_path: Union[str, Path]):
        self.nb_samples = self.file_lines(file_path=file_path)

        self.f = open(file_path, 'r')
        atexit.register(self.cleanup)

    def cleanup(self):
        if not self.f.closed:
            self.f.close()

    @classmethod
    def file_lines(cls, file_path: Union[str, Path]) -> int:
        with open(file_path, 'rb') as f:
            bufgen = iter(partial(f.raw.read, 1024 * 1024), b'')
            return sum(buf.count(b'\n') for buf in bufgen)

    def __len__(self) -> int:
        return self.nb_samples
