#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import os
import logging

import numpy as np
import tensorflow as tf

from daikon import reader
from daikon import constants as C
from daikon.vocab import create_vocab
from daikon.compgraph import define_computation_graph


def train(source_data: str, target_data: str, epochs: int, batch_size: int, vocab_max_size: int,
          save_to: str, log_to: str, **kwargs):
    """Trains a language model. See argument description in `bin/romanesco`."""

    # create folders for model and logs if they don't exist yet
    for folder in [save_to, log_to]:
        if not os.path.exists(folder):
            os.makedirs(folder)

    # create vocabulary to map words to ids, for source and target
    source_vocab = create_vocab(source_data, vocab_max_size, save_to, C.SOURCE_VOCAB_FILENAME)
    target_vocab = create_vocab(target_data, vocab_max_size, save_to, C.TARGET_VOCAB_FILENAME)

    # convert training data to list of word ids
    reader_ids = reader.read_parallel(source_data, target_data, source_vocab, target_vocab, C.MAX_LEN)

    # define computation graph
    logging.info("Building computation graph.")

    graph_components = define_computation_graph(source_vocab.size, target_vocab.size, batch_size)
    encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, _, _, summary = graph_components

    saver = tf.train.Saver()

    with tf.Session() as session:
        # init
        session.run(tf.global_variables_initializer())
        # write logs (@tensorboard)
        summary_writer = tf.summary.FileWriter(log_to, graph=tf.get_default_graph())

        logging.info("Starting training.")

        # iterate over training data `epochs` times
        for epoch in range(1, epochs + 1):
            total_loss = 0.0
            total_iter = 0
            for x, y, z in reader.iterate(reader_ids, batch_size, shuffle=True):

                feed_dict = {encoder_inputs: x,
                             decoder_inputs: y,
                             decoder_targets: z}

                l, _, s = session.run([loss, train_step, summary],
                                      feed_dict=feed_dict)
                summary_writer.add_summary(s, total_iter)
                total_loss += l
                total_iter += 1
                if total_iter % 10 == 0:
                    logging.debug("Epoch=%s, iteration=%s", epoch, total_iter)
            perplexity = np.exp(total_loss / total_iter)
            logging.info("Perplexity on training data after epoch %s: %.2f", epoch, perplexity)
            saver.save(session, os.path.join(save_to, C.MODEL_FILENAME))
