#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import tensorflow as tf

from daikon import constants as C
try:
    from tensorflow.python.ops.rnn import rnn_cell_impl as _rnn_cell, dynamic_rnn as _drnn, static_rnn as _rnn, static_bidirectional_rnn as _brnn
    from tensorflow.python.ops.rnn import rnn_cell_impl as _rnn_cell, dynamic_rnn as _drnn, static_rnn as _rnn, bidirectional_dynamic_rnn as _brnn
except:
    pass
def compute_lengths(sequences):
    """
    This solution is similar to:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.abs(sequences))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths


def define_computation_graph(source_vocab_size: int, target_vocab_size: int, batch_size: int):

    tf.reset_default_graph()

    # Placeholders for inputs and outputs
    encoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='encoder_inputs')

    decoder_targets = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')

    with tf.variable_scope("Embeddings"):
        source_embedding = tf.get_variable('source_embedding', [source_vocab_size, C.EMBEDDING_SIZE])
        target_embedding = tf.get_variable('target_embedding', [source_vocab_size, C.EMBEDDING_SIZE])

        encoder_inputs_embedded = tf.nn.embedding_lookup(source_embedding, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(target_embedding, decoder_inputs)

    with tf.variable_scope("Encoder"):
        fw_encoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
        bw_encoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)

        fw_encoder_cell=tf.contrib.rnn.DropoutWrapper(fw_encoder_cell,output_keep_prob=0.2)
        bw_encoder_cell=tf.contrib.rnn.DropoutWrapper(fw_encoder_cell,output_keep_prob=0.2)

        fw_initial_state = fw_encoder_cell.zero_state(batch_size, tf.float32)
        bw_initial_state = bw_encoder_cell.zero_state(batch_size, tf.float32)
        encoder_outputs, encoder_final_state = _brnn(cell_fw=fw_encoder_cell,cell_bw=bw_encoder_cell,
                                                                 inputs=encoder_inputs_embedded,
                                                                 initial_state_fw=fw_initial_state,initial_state_bw=bw_initial_state,
                                                                 dtype=tf.float32)

    with tf.variable_scope("Decoder"):
        decoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
        decoder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,output_keep_prob=0.2)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(cell=decoder_cell,
                                                                 inputs=decoder_inputs_embedded,initial_state=encoder_outputs,
                                                                 dtype=tf.float32)

    with tf.variable_scope("Additional_Coder"):
        coder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
        coder_cell = tf.contrib.rnn.DropoutWrapper(decoder_cell,output_keep_prob=0.2)

        coder_outputs, coder_final_state = tf.nn.dynamic_rnn(cell=coder_cell,
                                                                inputs=coder_inputs_embedded,initial_state=decoder_outputs,
                                                                 dtype=tf.float32)

with tf.variable_scope("Additional_Coder2"):
        coder_cell2 = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
        coder_cell2 = tf.contrib.rnn.DropoutWrapper(decoder_cell,output_keep_prob=0.2)

        coder_outputs2, decoder_final_state2 = tf.nn.dynamic_rnn(cell=coder_cell2,
                                                                inputs=coder_inputs2_embedded,initial_state=coder_outputs,
                                                                 dtype=tf.float32)

    with tf.variable_scope("Logits"):
        decoder_logits = tf.contrib.layers.linear(coder_outputs2, target_vocab_size)

    with tf.variable_scope("Loss"):
        one_hot_labels = tf.one_hot(decoder_targets, depth=target_vocab_size, dtype=tf.float32)
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=decoder_logits)

        # mask padded positions
        target_lengths = compute_lengths(decoder_targets)
        target_weights = tf.sequence_mask(lengths=target_lengths, maxlen=None, dtype=decoder_logits.dtype)
        weighted_cross_entropy = stepwise_cross_entropy * target_weights
        loss = tf.reduce_mean(weighted_cross_entropy)

    with tf.variable_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(learning_rate=C.LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, decoder_logits, summary
