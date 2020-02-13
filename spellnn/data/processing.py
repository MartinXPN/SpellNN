import random
from typing import List, Tuple, Optional

import numpy as np
import spacy
from keras_preprocessing.sequence import pad_sequences
from sacremoses import MosesDetokenizer

from spellnn.data import alphabet
from spellnn.data.spelling_mistakes import apply_spelling_errors, Mistakes


class DataProcessor:
    def __init__(self, locale: str, char2id,
                 alphabet: List[str], alphabet_weighs: Optional[List[float]] = None,
                 cache_limit: int = 1e6):
        self.nlp = spacy.load("en_core_web_sm", disable=['tagger', 'parser', 'textcat'])
        print('Initialized SpaCY with pipes:', self.nlp.pipe_names)
        self.detokenize = MosesDetokenizer(lang=locale).detokenize
        self.batch_docs = {}
        self.char2id = char2id
        self.alphabet = alphabet
        self.alphabet_weighs = alphabet_weighs
        self.mistakes = Mistakes(alphabet=alphabet, weights=alphabet_weighs)
        self.cache_limit = cache_limit

    def to_sample(self, line: str) -> Tuple[List[str], List[str], List[str]]:
        nb_words = random.randint(5, 30)
        doc = self.batch_docs[line] if line in self.batch_docs else self.nlp(line)
        words = doc[:nb_words]
        target = self.detokenize([t.text for t in words])

        encoder_inputs = list(apply_spelling_errors(words.text, mistakes=self.mistakes))
        decoder_inputs = [alphabet.START] + list(target)
        targets = list(target) + [alphabet.END]
        return encoder_inputs, decoder_inputs, targets

    def process_input(self, line: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        encoder_inputs, decoder_inputs, targets = self.to_sample(line)
        return self.char2id[encoder_inputs], self.char2id[decoder_inputs], self.char2id[targets]

    def doc_to_spans(self, texts: List[str], join_string: str = ' ||| ') -> List:
        all_docs = self.nlp(join_string.join(texts))
        split_ids = [i for i, token in enumerate(all_docs) if token.text == join_string.strip()] + [len(all_docs)]
        new_docs = [all_docs[(i + 1 if i > 0 else i): j] for i, j in zip([0] + split_ids[:-1], split_ids)]
        return new_docs

    def process_batch(self, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        lines = [line.decode('utf-8') for line in b]
        if len(self.batch_docs) > self.cache_limit:
            self.batch_docs = {}

        unprocessed_lines = [line for line in lines if line not in self.batch_docs]
        self.batch_docs.update(zip(unprocessed_lines, self.doc_to_spans(unprocessed_lines)))

        batch = np.array([self.process_input(line) for line in lines])
        encoder_batch, decoder_batch, target_batch = batch[:, 0, ...], batch[:, 1, ...], batch[:, 2, ...]
        return (pad_sequences(encoder_batch, padding='post', truncating='post', value=self.char2id[alphabet.END]),
                pad_sequences(decoder_batch, padding='post', truncating='post', value=self.char2id[alphabet.END]),
                pad_sequences(target_batch, padding='post', truncating='post', value=self.char2id[alphabet.END]))
