import logging
import os
from datetime import datetime
from inspect import signature, Parameter
from pathlib import Path
from pprint import pprint
from textwrap import dedent
from typing import Optional, Union

import fire
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, TerminateOnNaN
from tensorflow.keras import Model

from spellnn import models
from spellnn.data import alphabet
from spellnn.data.alphabet import get_chars
from spellnn.data.processing import DataProcessor
from spellnn.data.util import nb_lines
from spellnn.layers.mapping import CharMapping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class Gym:
    def __init__(self):
        self.train_dataset: Optional[tf.data.Dataset] = None
        self.valid_dataset: Optional[tf.data.Dataset] = None
        self.char2int: Optional[CharMapping] = None
        self.model: Optional[Model] = None
        self.nb_train_samples: int = 0
        self.nb_valid_samples: int = 0
        self.batch_size = 0

    def construct_dataset(self, path: str, locale: str, batch_size: int = 32, validation_split: float = 0.3):
        pprint(locals())
        all_chars = [alphabet.START + alphabet.END] + get_chars(locale)
        char_weights = [0.5 if c.isalpha() and c.islower() else
                        0.2 if c.isalpha() else
                        0.1 if c not in {alphabet.START, alphabet.END} else
                        0 for c in all_chars]
        self.char2int = CharMapping(chars=all_chars, include_unknown=True)
        data_processor = DataProcessor(locale=locale, char2id=self.char2int,
                                       alphabet=all_chars, alphabet_weighs=char_weights)

        print('Calculating number of lines in the file...', end=' ')
        all_samples = nb_lines(path)
        print(all_samples)

        self.batch_size = batch_size
        self.nb_train_samples = int((1 - validation_split) * all_samples)
        self.nb_valid_samples = all_samples - self.nb_train_samples

        dataset = tf.data.TextLineDataset(path)
        self.train_dataset = dataset.take(self.nb_train_samples)
        self.train_dataset = self.train_dataset.shuffle(10 * batch_size, seed=42, reshuffle_each_iteration=True)
        self.train_dataset = self.train_dataset.batch(batch_size, drop_remainder=True)
        self.train_dataset = self.train_dataset.map(
            lambda b: tf.numpy_function(func=data_processor.process_batch, inp=[b], Tout=['int32', 'int32', 'int32']))
        self.train_dataset = self.train_dataset.map(lambda enc_in, dec_in, targ: ((enc_in, dec_in), targ))
        self.train_dataset = self.train_dataset.repeat()

        self.valid_dataset = dataset.skip(self.nb_train_samples)
        self.valid_dataset = self.valid_dataset.shuffle(10 * batch_size, seed=42, reshuffle_each_iteration=True)
        self.valid_dataset = self.valid_dataset.batch(batch_size, drop_remainder=True)
        self.valid_dataset = self.valid_dataset.map(
            lambda b: tf.numpy_function(func=data_processor.process_batch, inp=[b], Tout=['int32', 'int32', 'int32']))
        self.valid_dataset = self.valid_dataset.map(lambda enc_in, dec_in, targ: ((enc_in, dec_in), targ))
        self.valid_dataset = self.valid_dataset.repeat()
        return self

    def create_model(self, name):
        arguments = signature(getattr(models, name).__init__)
        arguments = {k: v.default for k, v in arguments.parameters.items()
                     if v.default is not Parameter.empty and k != 'self'}
        arguments['nb_symbols'] = len(self.char2int)
        arg_str = ', '.join([f'{k}=' + str(v) if type(v) != str else f'{k}=' '"' + str(v) + '"'
                             for k, v in arguments.items()])
        # print(arg_str)
        exec(dedent(f'''
        def create({arg_str}):
            self.model = {name}(**locals())
            return self
        create.__name__ = {name}.__name__
        create.__doc__ = {name}.__init__.__doc__
        setattr(self, create.__name__, create)
        '''), {'self': self, name: getattr(models, name), arg_str: arg_str})
        return getattr(self, name)

    def train(self, epochs: int, monitor_metric='val_acc', patience: int = 5,
              steps_per_epoch: Union[int, str] = 'auto', validation_steps: Union[int, str] = 'auto',
              log_dir: str = 'logs',
              use_multiprocessing: bool = False):
        pprint(locals())
        log_dir = Path(log_dir).joinpath(datetime.now().replace(microsecond=0).isoformat())
        model_path = Path(log_dir).joinpath('checkpoints').joinpath('best-model.h5py')
        model_path = str(model_path)

        if steps_per_epoch == 'auto':
            steps_per_epoch = self.nb_train_samples // self.batch_size
        if validation_steps == 'auto':
            validation_steps = self.nb_valid_samples // self.batch_size

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
        history = self.model.fit_generator(
            self.train_dataset.as_numpy_iterator(), steps_per_epoch=steps_per_epoch,
            validation_data=self.valid_dataset.as_numpy_iterator(), validation_steps=validation_steps,
            epochs=epochs,
            use_multiprocessing=use_multiprocessing, workers=os.cpu_count() - 1,
            callbacks=[
                TerminateOnNaN(),
                TensorBoard(log_dir=log_dir),
                ModelCheckpoint(model_path, monitor=monitor_metric, verbose=1, save_best_only=True),
                EarlyStopping(monitor=monitor_metric, patience=patience),
            ])
        return history.history


if __name__ == '__main__':
    cli = Gym()
    fire.Fire(cli)
