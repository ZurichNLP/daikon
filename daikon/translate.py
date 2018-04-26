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

    # fix batch_size to 1 for now
    encoder_inputs, decoder_targets, decoder_inputs, _, _, decoder_logits, _ = compgraph.define_computation_graph(source_vocab.size, target_vocab.size, 1)

    saver = tf.train.Saver()

    source_ids = np.array(source_vocab.get_ids(input_text.split())).reshape(1, -1)

    # target ids will serve as decoder inputs and decoder targets
    # TODO: increase permissible length of translation?
    target_ids = np.full(shape=(1, C.MAX_LEN), fill_value=C.UNK_ID, dtype=np.int)

    with tf.Session() as session:
        # TODO: needed?
        # session.run(tf.global_variables_initializer())

        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        sampled_sequence = []

        for index in range(C.MAX_LEN):

            feed_dict = {encoder_inputs: source_ids,
                         decoder_inputs: target_ids,
                         decoder_targets: target_ids}
            logits_result = session.run([decoder_logits], feed_dict=feed_dict)

            next_symbol_logits = logits_result[0][0][index]
            next_symbol_probs = softmax(next_symbol_logits)

            sampled_symbol = np.random.choice(range(target_vocab.size), p=next_symbol_probs)

            if sampled_symbol in [C.EOS_ID, C.PAD_ID]:
                break

            sampled_sequence.append(sampled_symbol)
            target_ids[0][index] = sampled_symbol

    words = target_vocab.get_words(sampled_sequence)
    return ' '.join(words)
