import abc
import atexit
import random
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import List, Iterator, overload, Union, Dict

from spellnn.entities import Sample


class Dataset:
    """
    Abstract class for a dataset.
    Any subclass implementing the abstract methods of this class will work within SpellNN
    """
    @abc.abstractmethod
    def __len__(self) -> int:
        """
        :return: number of samples in the dataset
        """

    def __iter__(self) -> Iterator[Sample]:
        """
        :return: iter(self.data)
        """
        return iter(self.DataIterator(self))

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

    class DataIterator:
        def __init__(self, dataset):
            self.i = 0
            self.dataset = dataset

        def __iter__(self) -> Iterator[Sample]:
            """
            :return: iter(self.data)
            """
            return self

        def __next__(self):
            if self.i > len(self.dataset):
                raise StopIteration
            else:
                self.i += 1
                return self.dataset[self.i - 1]


class PlainTextFileDataset(Dataset):

    def __init__(self, file_path: Union[str, Path]):
        super().__init__()
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


class LargeTextFileDataset(Dataset):
    """
    As plain text files can be large and reach up to several GB in storage,
    this class implements lazy loading of the dataset file.
    """

    def __init__(self, file_path: Union[str, Path], max_samples: int = 1024 * 128, read_chunk_samples: int = 1024 * 32):
        """
        :param file_path: path to dataset file
        :param max_samples: maximum number of samples to be kept in the memory
        :param read_chunk_samples: how many samples to read while loading the next chunk
        """
        super().__init__()
        self.nb_samples = self.file_lines(file_path=file_path)
        self.max_samples = max_samples
        self.read_chunk_size = read_chunk_samples
        assert max_samples >= read_chunk_samples
        self.current_id: int = 0
        self.id2sample: Dict[int, Sample] = {}
        self.time2id: OrderedDict[float, int] = OrderedDict()

        self.return_shuffled = False

        self.f = open(file_path, 'r', encoding='utf-8')
        self.file_path = file_path
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

    def load_chunk(self):
        for i in range(self.read_chunk_size):
            line = self.f.readline().strip()
            if not line:
                self.f.close()
                self.f = open(self.file_path, 'r', encoding='utf-8')
                self.current_id = 0
                line = self.f.readline().strip()

            while len(self.time2id) > self.max_samples or self.current_id in self.id2sample:
                t, index = self.time2id.popitem(last=False)
                del self.id2sample[index]
            assert len(self.id2sample) == len(self.time2id)

            self.time2id[time.time()] = self.current_id
            self.id2sample[self.current_id] = Sample(text=line)
            self.current_id += 1
        print('Chunk loaded:')
        for sample in self.id2sample.values():
            print(sample.text)

    def shuffle(self):
        self.return_shuffled = True

    @overload
    def __getitem__(self, i: slice) -> List[Sample]:
        ...

    @overload
    def __getitem__(self, i: int) -> Sample:
        ...

    def __getitem__(self, i):
        if isinstance(i, int):
            while i not in self.id2sample:
                self.load_chunk()
            return self.id2sample[i]

        if isinstance(i, slice):
            res = []
            for idx in range(i.start, i.stop, i.step):
                res.append(self[idx])
            return res
