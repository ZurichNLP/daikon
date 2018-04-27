#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

PAD_ID = 0
EOS_ID = 1
BOS_ID = EOS_ID
UNK_ID = 2

PAD = '<pad>'
EOS = '<eos>'
BOS = EOS
UNK = '<unk>'

MODEL_FILENAME = 'model'
SOURCE_VOCAB_FILENAME = 'vocab.source.json'
TARGET_VOCAB_FILENAME = 'vocab.target.json'

# max number of tokens per sequence
MAX_LEN = 50
SOURCE_VOCAB_SIZE = 10000
TARGET_VOCAB_SIZE = 10000

EMBEDDING_SIZE = 512
# size of LSTM hidden state vectors
HIDDEN_SIZE = 1024

NUM_LAYERS = 1
# truncate backpropagation though unrolled recurrent network
NUM_STEPS = MAX_LEN

SCORE_MAX_LEN = 1000
TRANSLATION_MAX_LEN = MAX_LEN * 2

LEARNING_RATE = 0.0001
