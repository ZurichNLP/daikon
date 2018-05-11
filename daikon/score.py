#!/usr/bin/env python3

import os

import numpy as np
import tensorflow as tf

from daikon import reader
from daikon import constants as C
from daikon.vocab import Vocabulary
from daikon.compgraph import define_computation_graph


def score(source_data: str, target_data: str, load_from: str, corpus_average: bool, normalize: bool, **kwargs):
    """Scores a text using a trained translation model. See argument description in `bin/daikon`."""

    # fix batch size at 1 to get individual scores for sentences
    batch_size = 1

    source_vocab = Vocabulary()
    target_vocab = Vocabulary()
    source_vocab.load(os.path.join(load_from, C.SOURCE_VOCAB_FILENAME))
    target_vocab.load(os.path.join(load_from, C.TARGET_VOCAB_FILENAME))

    reader_ids = list(reader.read_parallel(source_data, target_data, source_vocab, target_vocab, C.SCORE_MAX_LEN))

    encoder_inputs, decoder_targets, decoder_inputs, loss, _, _, _ = define_computation_graph(source_vocab.size, target_vocab.size, batch_size)

    saver = tf.train.Saver()

    with tf.Session() as session:
        # load model
        saver.restore(session, os.path.join(load_from, C.MODEL_FILENAME))

        losses = []
        total_iter = 0
        for x, y, z in reader.iterate(reader_ids, batch_size, shuffle=False):
            feed_dict = {encoder_inputs: x,
                         decoder_inputs: y,
                         decoder_targets: z}
            l = session.run([loss], feed_dict=feed_dict)

            # first session variable
            l = l[0]
            if normalize:
                # normalize by length of target sequence (including EOS token)
                l /= y.shape[1]

            losses.append(l)
            total_iter += 1

        if corpus_average:
            total_loss = np.sum(losses)
            perplexity = np.exp(total_loss / total_iter)
            return perplexity
        else:
            return np.exp(losses)
