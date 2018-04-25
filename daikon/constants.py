#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

PAD_ID = 0
EOS_ID = 1
UNK_ID = 2

EOS = '<eos>'
UNK = '<unk>'

# max number of tokens per sequence
MAX_LEN = 100
SOURCE_VOCAB_SIZE = 10000
TARGET_VOCAB_SIZE = 10000

EMBEDDING_SIZE = 128
# size of LSTM hidden state vectors
HIDDEN_SIZE = 512

NUM_LAYERS = 1
# truncate backpropagation though unrolled recurrent network
NUM_STEPS = MAX_LEN

LEARNING_RATE = 0.0001
