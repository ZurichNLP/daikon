#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import json
import os
import random
from typing import List
from collections import Counter

from daikon import constants as C
from daikon import reader


class Vocabulary:

    def __init__(self):
        self._id = {}  # {word: id}
        self._word = {}  # {id: word}

    def build(self, filename: str, max_size: int = None):
        """Builds a vocabulary mapping words (tokens) to ids (integers) and vice
        versa. The more frequent a word, the lower its id. 0 is reserved for
        unknown words.

        Args:
            filename: path to tokenised text file, one sentence per line.
            max_size: the maximum number of words (only keep most frequent n
                      words)
        """
        self.filename = filename
        self.max_size = max_size

        words = reader.read_words(filename)
        word_counts = Counter(words)
        sorted_words = [word for word, _ in word_counts.most_common() if word != C.UNK]
        # TODO: do not hard-code the id of special symbols like that
        sorted_words = [C.PAD, C.EOS, C.UNK] + sorted_words
        if max_size:
            sorted_words = sorted_words[:max_size]
        for i, word in enumerate(sorted_words):
            self._id[word] = i
            self._word[i] = word

    def load(self, filename: str):
        """Loads a vocabulary (saved with `self.save`)"""
        with open(filename) as f:
            for word, i in json.load(f).items():
                self._id[word] = i
                self._word[i] = word

    def __repr__(self):
        return "Vocabulary(filename=%s, size=%d, max_size=%d)" % (self.filename, self.size, self.max_size)

    @property
    def size(self):
        return len(self._id)

    def get_id(self, word: str):
        try:
            return self._id[word]
        except KeyError:
            return self._id[C.UNK]

    def get_word(self, id: int):
        return self._word[id]

    def get_ids(self, words: List[str]):
        return [self.get_id(word) for word in words]

    def get_words(self, ids: List[int]):
        return [self.get_word(id) for id in ids]

    def get_random_id(self):
        """Returns the id of a random word."""
        return random.choice(list(self._id.values()))

    def save(self, filepath):
        """Writes this vocabulary to a file in JSON format."""
        with open(filepath, 'w') as f:
            json.dump(self._id, f, indent=4)


def create_vocab(data: str, max_size: int, save_to: str, filename: str):
    vocab = Vocabulary()
    vocab.build(data, max_size=max_size)
    vocab.save(os.path.join(save_to, filename))

    return vocab
