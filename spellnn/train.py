import logging
import os
from inspect import signature, Parameter
from textwrap import dedent
from typing import Optional

import fire
import tensorflow as tf
from tensorflow.keras import Model

from spellnn import models
from spellnn.data.alphabet import get_chars
from spellnn.layers.mapping import CharMapping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)


class Gym:
    def __init__(self):
        self.train_dataset: Optional[tf.data.Dataset] = None
        self.valid_dataset: Optional[tf.data.Dataset] = None
        self.char2int: Optional[CharMapping] = None
        self.model: Optional[Model] = None

    def construct_dataset(self, path: str, locale: str, batch_size: int = 32, train_samples: int = 1000):
        self.char2int = CharMapping(chars=get_chars(locale), default='<PAD>')

    def create_model(self, name):
        arguments = signature(getattr(models, name).__init__)
        arguments = {k: v.default for k, v in arguments.parameters.items()
                     if v.default is not Parameter.empty and k != 'self'}
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

    def train(self, epochs: int, steps_per_epoch: int, validation_steps: int):
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = self.model.fit(self.train_dataset,
                                 epochs=epochs, steps_per_epoch=steps_per_epoch,
                                 validation_data=self.valid_dataset, validation_steps=validation_steps,
                                 verbose=1, workers=4, use_multiprocessing=True)
        return history.history


if __name__ == '__main__':
    cli = Gym()
    fire.Fire(cli)
