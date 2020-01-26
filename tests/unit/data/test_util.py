import unittest

from spellnn import get_project_root
from spellnn.data.util import nb_lines


class FileLineTest(unittest.TestCase):

    def setUp(self):
        self.sample_dataset_path = get_project_root() / 'datasets' / 'sample.txt'

    def test_nb_lines(self):
        self.assertEqual(nb_lines(self.sample_dataset_path), 50)
