#!/usr/bin/env python3

import os
import sys
import logging

import numpy as np
import tensorflow as tf

from typing import List

from daikon import vocab
from daikon import compgraph
from daikon import constants as C


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def translate(load_from: str, input_text: str = [], train_mode: bool = False, **kwargs):
    """Generates a translation by predicting from a trained translation model. See argument
    description in `bin/daikon`."""

    source_vocab = vocab.Vocabulary()
    source_vocab.load(os.path.join(load_from, C.SOURCE_VOCAB_FILENAME))
    target_vocab = vocab.Vocabulary()
    target_vocab.load(os.path.join(load_from, C.TARGET_VOCAB_FILENAME))

    # fix batch_size to 1
    encoder_inputs, decoder_targets, decoder_inputs, _, _, decoder_logits, _ = compgraph.define_computation_graph(source_vocab.size, target_vocab.size, 1)

    saver = tf.train.Saver()

    source_ids = np.array(source_vocab.get_ids(input_text.split())).reshape(1, -1)

    with tf.Session() as session:

        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        translated_sequence = []

        # TODO: increase permissible length of translation?
        for _ in range(C.MAX_LEN):

            # target ids will serve as decoder inputs and decoder targets,
            # but decoder targets will not be used to compute logits
            target_ids = np.array([C.BOS_ID] + translated_sequence).reshape(1, -1)

            feed_dict = {encoder_inputs: source_ids,
                         decoder_inputs: target_ids,
                         decoder_targets: target_ids}
            logits_result = session.run([decoder_logits], feed_dict=feed_dict)

            # first session result, first item in batch, target symbol at last position
            next_symbol_logits = logits_result[0][0][-1]
            next_symbol = np.max(next_symbol_logits)

            if next_symbol in [C.EOS_ID, C.PAD_ID]:
                break

            translated_sequence.append(next_symbol)

    words = target_vocab.get_words(translated_sequence)
    return ' '.join(words)
