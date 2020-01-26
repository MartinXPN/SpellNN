from pprint import pprint
from unittest import TestCase

from spellnn import get_project_root
from spellnn.train import Gym


class TestOverFitting(TestCase):
    def setUp(self):
        return
        self.trainer = Gym()
        self.batch_size = 32
        self.dataset_path = get_project_root() / 'datasets' / 'sample.txt'
        print(self.dataset_path)
        self.trainer.construct_dataset(path=str(self.dataset_path), locale='en',
                                       batch_size=self.batch_size, train_samples=50)
        self.trainer.valid_dataset = self.trainer.train_dataset
        self.trainer.create_model(name='Seq2SeqAttentionCNN')(nb_classes=100, hidden_units=(128, 64))

    def test_datasets(self):
        self.skipTest('Not addressed yet')
        for ex in self.trainer.train_dataset.take(2):
            print(len(ex), ex[0].numpy().shape, ex[1].numpy().shape)
            print(ex)
            self.assertEqual(ex[0].numpy().shape[0], ex[1].numpy().shape[0], 'Input and target batch size equality')
            self.assertGreaterEqual(self.batch_size, ex[0].numpy().shape[0],
                                    'Generate correct batch size (less for remainder)')

    def test_overfitting(self):
        self.skipTest('Not addressed yet')
        self.skipTest('Takes too long to test on Travis every time')
        history = self.trainer.train(epochs=10, steps_per_epoch=500, validation_steps=2)
        self.assertGreaterEqual(history['val_accuracy'][-1], 0.99)
        pprint(history)
