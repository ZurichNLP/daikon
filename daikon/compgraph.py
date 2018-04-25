#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import tensorflow as tf
from daikon import constants as C


def define_computation_graph(source_vocab_size: int, target_vocab_size: int, batch_size: int):

    # Placeholders for inputs and outputs
    encoder_inputs = tf.placeholder(shape=(batch_size, C.NUM_STEPS), dtype=tf.int32, name='encoder_inputs')

    decoder_targets = tf.placeholder(shape=(batch_size, C.NUM_STEPS), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(shape=(batch_size, C.NUM_STEPS), dtype=tf.int32, name='decoder_inputs')

    with tf.name_scope("Embeddings"):
        source_embeddings = tf.Variable(tf.random_uniform([source_vocab_size, C.EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32)
        target_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, C.EMBEDDING_SIZE], -1.0, 1.0), dtype=tf.float32)

        encoder_inputs_embedded = tf.nn.embedding_lookup(source_embeddings, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(target_embeddings, decoder_inputs)

    with tf.name_scope("Encoder"):
        encoder_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE)
        initial_state = encoder_cell.zero_state(batch_size, tf.float32)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                 encoder_inputs_embedded,
                                                                 initial_state=initial_state,
                                                                 dtype=tf.float32)

    with tf.name_scope("Decoder"):
        decoder_cell = tf.contrib.rnn.LSTMCell(C.HIDDEN_SIZE)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                 decoder_inputs_embedded,
                                                                 initial_state=encoder_final_state,
                                                                 dtype=tf.float32)

    with tf.name_scope("Logits"):
        decoder_logits = tf.contrib.layers.linear(decoder_outputs, target_vocab_size)

        # TODO: check axis is correct for batch-major
        decoder_prediction = tf.argmax(decoder_logits, axis=2)

    with tf.name_scope("Loss"):
        loss = tf.contrib.seq2seq.sequence_loss(logits=decoder_logits,
                                                targets=decoder_targets,
                                                # Using weights as a sequence mask
                                                weights=tf.ones([batch_size, C.NUM_STEPS]),
                                                average_across_timesteps=True,
                                                average_across_batch=True)

    with tf.name_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=C.LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, decoder_logits, decoder_prediction, summary
