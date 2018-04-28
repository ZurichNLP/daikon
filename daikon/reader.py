#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import random
import numpy as np
import tensorflow as tf

from typing import List, Tuple, Iterator

from daikon import constants as C


def read_words(filename: str) -> List[str]:
    """Reads a tokenised text.

    Args:
        filename: path to tokenised text file, one sentence per line.

    Returns:
        A single list for all tokens in all sentences, all newline
        characters replaced with spaces.
    """
    with tf.gfile.GFile(filename) as f:
        return f.read().replace("\n", " ").split()


def read_lines(filename: str) -> Iterator[List[str]]:
    """Reads a tokenised text file line by line.

    Args:
        filename: path to tokenised text file, one sentence per line.

    Returns:
        An iterator that yields lines one at a time, as a list of tokens.
    """
    with tf.gfile.GFile(filename) as f:
        for line in f:
            yield line.strip().split()


def read(filename: str, vocab) -> Iterator[List[int]]:
    """Turns a tokenised text into a list of token ids.

    Args:
        filename: path to tokenised text file, one sentence per line.
        vocab: an instance of type daikon.vocab.Vocabulary

    Returns:
        An iterator that yields a list of lists, where an individual
        list contains word ids for one input sentence.
    """
    lines = read_lines(filename)
    for line in lines:
        yield [vocab.get_id(word) for word in line]


def read_parallel(source_filename: str,
                  target_filename: str,
                  source_vocab,
                  target_vocab,
                  max_length: int):
    """
    Reads a pair of files that are parallel text.

    Args:
        source_filename: path to source text file.
        target_filename: path to target text file.
        source_vocab: an instance of type daikon.vocab.Vocabulary
        target_vocab: an instance of type daikon.vocab.Vocabulary
        max_length: if a sentence is longer than max_length, it
        is skipped.

    Returns:
        An iterator that yields tuples of two lists. The lists are
        the list of source word ids, and target word ids, respectively.
    """

    for source_ids, target_ids in zip(read(source_filename, source_vocab),
                                      read(target_filename, target_vocab)):
        if [] in [source_ids, target_ids]:
            # skip parallel segments where one or both sides are empty
            continue
        if (len(source_ids) > max_length) or (len(target_ids) > max_length):
            # skip segments that are too long
            continue

        yield (source_ids, target_ids)


def pad_sequence(word_ids: List[int], pad_id: int, max_length: int) -> List[int]:
    """
    Pads sequences if they are shorter than max_length.

    Args:
        word_ids: A list of word ids.
        pad_id: The ID that represents the padding symbol.
        max_length: The length up to which a sequence should be padded.

    Returns:
        A sequence that is padded to the right, if it was shorter than
        the maximum length.

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


def iterate(reader_ids: List[ReaderTuple], batch_size: int, shuffle: bool = True):
    """Yields padded sequences for NMT training (or translation),
    in batches of size `batch_size`.

    Args:
        reader_ids: A list of parallel pairs as returned by
                    daikon.reader.read_parallel
        batch_size: Number of sequences in a batch.
        shuffle: Whether the ordering of sequences should be random.

    Returns:
        An iterator that returns encoder inputs, decoder inputs and decoder
        targets, in batches, as padded numpy arrays.

    Example:
        >>> from daikon import reader
        >>> reader_ids = [([1,2,3,4], [5,6,7,8,9]), ([10, 11], [12, 13, 14, 15]), ([16, 17, 18], [19]), ([20, 21], [22, 23]), ([24, 25], [26, 27]), ([28, 29], [30, 31])]
        >>> batches = list(reader.iterate(reader_ids, batch_size=2, shuffle=False))
        >>> len(batches)
        3
        >>> encoder_inputs, decoder_inputs, decoder_targets = batches[0]
        >>> encoder_inputs
        array([[ 1,  2,  3,  4],
               [10, 11,  0,  0]])
        >>> decoder_inputs
        array([[ 1,  5,  6,  7,  8,  9],
               [ 1, 12, 13, 14, 15,  0]])
        >>> decoder_targets
        array([[ 5,  6,  7,  8,  9,  1],
               [12, 13, 14, 15,  1,  0]])
    """
    if shuffle:
        reader_ids = list(reader_ids)  # TODO: do not make actual copies here
        random.shuffle(reader_ids)

    source_data_ids, target_data_ids = zip(*reader_ids)

    source_data_ids = list(source_data_ids)
    target_data_ids = list(target_data_ids)

    source_len = len(source_data_ids)
    target_len = len(target_data_ids)

    assert source_len == target_len, "Source and target do not have the same number of segments."

    num_batches = source_len // batch_size

    source_data_ids = source_data_ids[0: batch_size * num_batches]  # cut off
    target_data_ids = target_data_ids[0: batch_size * num_batches]  # cut off

    for i in range(num_batches):
        s = i * batch_size
        e = s + batch_size

        source_slice = source_data_ids[s:e]
        target_slice = target_data_ids[s:e]

        # add EOS and BOS symbols
        decoder_targets = [sequence + [C.EOS_ID] for sequence in target_slice]
        # shifted by one token to the right
        decoder_inputs = [[C.BOS_ID] + sequence for sequence in target_slice]

        max_len_in_source = max([len(s) for s in source_slice])
        max_len_in_target = max([len(s) for s in decoder_targets])

        encoder_inputs = [pad_sequence(ids, C.PAD_ID, max_len_in_source) for ids in source_slice]
        decoder_inputs = [pad_sequence(ids, C.PAD_ID, max_len_in_target) for ids in decoder_inputs]
        decoder_targets = [pad_sequence(ids, C.PAD_ID, max_len_in_target) for ids in decoder_targets]

        yield np.array(encoder_inputs), np.array(decoder_inputs), np.array(decoder_targets)
