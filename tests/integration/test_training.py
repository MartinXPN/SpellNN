from pprint import pprint
from unittest import TestCase

from spellnn import get_project_root
from spellnn.train import Gym


class TestOverFitting(TestCase):
    def setUp(self):
        self.trainer = Gym()
        self.batch_size = 8
        self.dataset_path = get_project_root() / 'datasets' / 'sample.txt'
        print(self.dataset_path)
        self.trainer.construct_dataset(path=str(self.dataset_path), locale='en',
                                       batch_size=self.batch_size, validation_split=0.3)
        self.trainer.create_model(name='Seq2SeqAttentionCNN')(nb_symbols=len(self.trainer.char2int), embedding_size=8)

    def test_datasets(self):

        def print_first(dataset, k: int):
            for ex in dataset.take(k):
                print(len(ex), len(ex[0]), len(ex[1]))
                print(ex[0][0].numpy().shape)
                print(ex[0][1].numpy().shape)
                print(ex[1].numpy().shape)

        print_first(self.trainer.train_dataset, 5)
        print_first(self.trainer.valid_dataset, 5)

        for sample in self.trainer.train_dataset.take(2):
            print(sample)
            self.assertGreaterEqual(self.batch_size, sample[1].numpy().shape[0],
                                    'Generate correct batch size (less for remainder)')

        # Make sure we can iterate over the dataset as many times as we want
        for _ in self.trainer.train_dataset.take(150):
            pass

    def test_training_once(self):
        history = self.trainer.train(epochs=2)
        pprint(history)

    def test_overfitting(self):
        self.skipTest('Not addressed yet')
        self.skipTest('Takes too long to test on Travis every time')
        history = self.trainer.train(epochs=10)
        self.assertGreaterEqual(history['val_accuracy'][-1], 0.99)
        pprint(history)
