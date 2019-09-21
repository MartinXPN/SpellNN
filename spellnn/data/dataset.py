import abc
import atexit
import math
import random
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
            for line in f:
                line = line.strip()
                if line != '':
                    self.data.append(Sample(line))

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
    def __init__(self, file_path: str):
        self.f = open(file_path, 'r')
        self.sample_count = 0
        self.processed_once = False

        atexit.register(self.cleanup)

    def cleanup(self):
        self.f.close()

    def __len__(self) -> int:
        """
        __len__ is infinity before reaching the end of file for the first time.
        After that, the number of samples is fixed
        :return: number of samples in the dataset
        """
        return self.sample_count if self.processed_once else math.inf
