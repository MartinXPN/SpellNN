import random
import string
from pathlib import Path
from textwrap import dedent
from unittest import TestCase

from spellnn.data.dataset import PlainTextFileDataset


class TestDataset(TestCase):

    def setUp(self):
        super().setUp()
        self.file_path = None

    def test_small_file_loading(self):
        self.file_path = Path('small.txt')
        lines = [u'Line one', u'Line two,', u'Третья строка!', u'Línea tres.']

        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(dedent(f'''
            {lines[0]}
            
            {lines[1]}
            {lines[2]}
            
            
            {lines[3]}'''))

        dataset = PlainTextFileDataset(file_path=self.file_path)
        self.assertEqual(len(dataset), len(lines))
        for i in range(len(lines)):
            self.assertEqual(dataset[i].text, lines[i])

    def test_file_dataset_iteration(self):
        self.file_path = Path('iteration.txt')
        lines = [u'Line one', u'Line two,', u'Третья строка!', u'Línea tres.']

        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write(dedent(f'''
            {lines[0]}

            {lines[1]}
            {lines[2]}


            {lines[3]}'''))

        dataset = PlainTextFileDataset(file_path=self.file_path)
        self.assertEqual(len(dataset), len(lines))
        for sample, line in zip(dataset, lines):
            self.assertEqual(sample.text, line)

        dataset.shuffle()

    def test_shuffle(self):
        self.file_path = Path('shuffle.txt')
        lines = [''.join(random.choices(string.printable.replace('\n', '').strip(), k=random.randint(1, 20)))
                 for _ in range(15)]

        with open(self.file_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        dataset = PlainTextFileDataset(file_path=self.file_path)
        dataset_lines = [sample.text for sample in dataset]
        self.assertEqual(len(dataset), len(lines), f'{dataset_lines} != {lines}')
        self.assertListEqual(dataset_lines, lines, f'{dataset_lines} != {lines}')

        dataset.shuffle()
        dataset_lines = [sample.text for sample in dataset]
        self.assertCountEqual(dataset_lines, lines, f'{dataset_lines} != {lines}')
        self.assertFalse(all([dataset_line == line for dataset_line, line in zip(dataset_lines, lines)]))

    def doCleanups(self):
        super().doCleanups()
        if self.file_path and self.file_path.exists():
            self.file_path.unlink()
