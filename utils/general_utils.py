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


def load_word_embedding_dict(embedding, embedding_path, word_alphabet, logger, embedd_dim=100):
    """
    load word embeddings from file
    :param embedding:
    :param embedding_path:
    :param logger:
    :return: embedding dict, embedding dimention, caseless
    """
    if embedding == 'glove':
        # loading GloVe
        logger.info("Loading GloVe ...")
        embedd_dim = -1
        embedd_dict = dict()
        # with gzip.open(embedding_path, 'r') as file:
        with open(embedding_path, 'r') as file:
            for line in file:
                line = line.strip()
                if len(line) == 0:
                    continue

                tokens = line.split()
                if embedd_dim < 0:
                    embedd_dim = len(tokens) - 1
                else:
                    assert (embedd_dim + 1 == len(tokens))
                embedd = np.empty([1, embedd_dim], dtype=theano.config.floatX)
                embedd[:] = tokens[1:]
                embedd_dict[tokens[0]] = embedd
        return embedd_dict, embedd_dim, True
    else:
        raise ValueError("embedding should choose from [glove, ...]")


def build_embedd_table(word_alphabet, embedd_dict, embedd_dim, logger, caseless):
    scale = np.sqrt(3.0 / embedd_dim)
    embedd_table = np.empty([len(word_alphabet), embedd_dim], dtype=theano.config.floatX)
    embedd_table[0, :] = np.zeros([1, embedd_dim])
    oov_num = 0
    # for word, index in word_alphabet.iteritems():
    for word in word_alphabet:
        ww = word.lower() if caseless else word
        # show oov ratio
        if ww in embedd_dict:
            embedd = embedd_dict[ww]
        else:
            embedd = np.random.uniform(-scale, scale, [1, embedd_dim])
            oov_num += 1
        b = word_alphabet[word]
        embedd_table[word_alphabet[word], :] = embedd
    oov_ratio = float(oov_num)/(len(word_alphabet)-1)
    logger.info("OOV number =%s, OOV ratio = %f" % (oov_num, oov_ratio))
    return embedd_table