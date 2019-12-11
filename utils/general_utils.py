import logging
import sys
import numpy as np
import theano


def get_logger(name, level=logging.INFO, handler=sys.stdout,
        formatter='%(name)s - %(levelname)s - %(message)s'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(formatter)
    stream_handler = logging.StreamHandler(handler)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def padding_email_sequences(index_sequences, labels, max_seq_length, post_padding=True):

    X = np.empty([len(index_sequences), max_seq_length], dtype=np.int32)
    Y = np.empty([len(index_sequences), 1], dtype=np.float32)
    mask = np.zeros([len(index_sequences), max_seq_length], dtype=theano.config.floatX)

    for i in range(len(index_sequences)):
        single_essay_word_ids = index_sequences[i]
        num = len(single_essay_word_ids)

        for j in range(num):
            word_id = single_essay_word_ids[j]
            X[i, j] = word_id

            # Zero out X after the end of the sequence
            X[i, num:] = 0
            # Make the mask for this sample 1 within the range of length
            mask[i, :num] = 1

        Y[i] = labels[i]
    return X, Y, mask