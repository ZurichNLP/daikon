#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import numpy as np
import tensorflow as tf

from random import shuffle
from typing import List, Tuple

from daikon import constants as C
from daikon import vocab


def read_words(filename: str):
    """Reads a tokenised text.

    Args:
        filename: path to tokenised text file, one sentence per line.

    Returns:
        A single list for all tokens in all sentences, with sentence boundaries
        indicated by <eos> (end of sentence).
    """
    with tf.gfile.GFile(filename) as f:
        return f.read().replace("\n", " " + C.EOS + " ").split()


def read_lines(filename: str):
    """Reads a tokenised text.

    Args:
        filename: path to tokenised text file, one sentence per line.

    Returns:
        An iterator that yields lines one at a time, as a list of tokens.
    """
    with tf.gfile.GFile(filename) as f:
        for line in f:
            yield line.strip().split() + C.EOS


def read(filename: str, vocab: vocab.Vocabulary):
    """Turns a tokenised text into a list of token ids.

    Args:
        filename: path to tokenised text file, one sentence per line.
        vocab: an instance of type romanesco.vocab.Vocabulary

    Returns:
        A list of lists, where an individual list contains word ids for
        one input sentence.
    """
    lines = read_lines(filename)
    for line in lines:
        yield [vocab.get_id(word) for word in line]


def read_parallel(source_filename: str,
                  target_filename: str,
                  source_vocab: vocab.Vocabulary,
                  target_vocab: vocab.Vocabulary,
                  max_length: int):

    for source_ids, target_ids in zip(read(source_filename, source_vocab),
                                      read(target_filename, target_vocab)):
        if [] in [source_ids, target_ids]:
            # skip parallel segments where one or both sides are empty
            continue
        if (len(source_ids) > max_length) or (len(target_ids) > max_length):
            # skip segments that are too long
            continue

        yield (source_ids, target_ids)


def pad_sequence(word_ids: List[int], pad_id: int, max_length: int):
    """
    Pads sequences if they are shorter than max_length.

    Example:
    >>> pad_sequence([1, 2, 3, 4], pad_id=0, max_length=10)
    array([1, 2, 3, 4, 0, 0, 0, 0, 0, 0])
    """
    padded_sequence = np.full(shape=(max_length,), fill_value=pad_id, dtype=np.int)

    for index, word_id in enumerate(word_ids):
        padded_sequence[index] = word_id

    return padded_sequence


NestedIds = List[List[int]]
ReaderTuple = Tuple[NestedIds, NestedIds]


def iterate(reader_ids: ReaderTuple, batch_size: int):
    """Yields sequences of length `num_steps` for NMT training (or translation),
    in batches of size `batch_size`.

    Args:
        raw_data: the dataset (a list of numbers).
        batch_size: the batch size
        num_steps: number of time steps per example

    Yields:
        an (x, y, z) tuple, with x corresponding to inputs and y to expected
        outputs. y is x time shifted by one: y_0 = x_1, y_1 = x_2, etc. Both x
        and y are NumPy arrays of shape (num_steps, batch_size).

    Example:
        >>> raw_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        >>> i = iterator(raw_data, batch_size=3, num_steps=2)
        >>> batches = list(i)
        >>> len(batches)
        2
        >>> batches[0]
        ( [[ 0,  1],  [[ 1,  2],
           [ 5,  6],   [ 6,  7],
           [10, 11]],  [11, 12]] )
        >>> batches[1]
        ( [[ 2,  3],  [[ 3,  4],
           [ 7,  8],   [ 8,  9],
           [12, 13]]   [13, 14]] )
    """
    shuffle(reader_ids)

    source_data_ids, target_data_ids = zip(*reader_ids)

    source_len = len(source_data_ids)
    target_len = len(target_data_ids)

    assert source_len == target_len, "Source and target do not have the same number of segments."

    num_batches = source_len // batch_size

    source_data_ids = source_data_ids[0 : batch_size * num_batches]  # cut off
    target_data_ids = target_data_ids[0 : batch_size * num_batches]  # cut off

    source_data_ids = np.reshape(source_data_ids, (batch_size, num_batches))
    target_data_ids = np.reshape(target_data_ids, (batch_size, num_batches))



    # raw_data = [the cat sits on the mat and eats a tasty little tuna fish .]
    # data = [[the cat   sits   on  ]
    #         [the mat   and    eats]
    #         [a   tasty little tuna]]  with batch_size = 3

    num_batches_in_epoch = (num_batches - 1) // num_steps
    # -1 because y will be x, time shifted by 1

    for i in range(num_batches_in_epoch):
        s = i * num_steps # start
        e = s + num_steps # end
        yield data[:, s : e], data[:, s + 1 : e + 1]

        # ( [[the cat  ],  [[cat   sits],
        #    [the mat  ],   [mat   and ],
        #    [a   tasty]],  [tasty little]] )
