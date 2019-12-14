from time import time
import keras.backend as K
import numpy as np
from sklearn.metrics import accuracy_score
from keras.preprocessing.sequence import pad_sequences
from utils.metrics import precision, recall, f1_score, false_positive_rate
from utils.clean_data import process_legit_phish_data
from utils.preprocess_data import split_data, extract_labels, create_vocab, read_dataset, create_char_vocab, read_dataset_word_char
from utils.general_utils import get_logger, padding_email_sequences, load_word_embedding_dict, build_embedd_table
from models.themis_models import build_simple_themis, build_simple_themis_word_char
from evaluators.evaluator_word_char import Evaluator

logger = get_logger("Train ...")


def main():
    all_data = process_legit_phish_data(legit_path='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/legit/', phish_path='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/phish/')
    embedding_path = 'embeddings/glove.6B.50d.txt'
    embedding = 'glove'
    embedd_dim = 50
    epochs = 20
    batch_size = 16
    baby = True
    if baby:
        all_data = all_data[:100]
    train, dev, test = split_data(all_data)
    x_train, y_train = extract_labels(train)
    x_dev, y_dev = extract_labels(dev)
    x_test, y_test = extract_labels(test)
    vocab = create_vocab(x_train, vocab_size=20000, to_lower=True)
    
    char_vocab = create_char_vocab(x_train, to_lower=True)
    x_train, x_train_char, max_token_train, max_char_token_train = read_dataset_word_char(x_train, vocab, char_vocab, to_lower=True)
    x_dev, x_dev_char, max_token_dev, max_char_token_dev = read_dataset_word_char(x_dev, vocab, char_vocab, to_lower=True)
    x_test, x_test_char, max_token_test, max_char_token_test = read_dataset_word_char(x_test, vocab, char_vocab, to_lower=True)
    
    max_char_token = max(max_char_token_train, max_char_token_dev, max_char_token_test)
    if max_char_token > 300:
        max_char_token = 300

    max_token = max(max_token_train, max_token_dev, max_token_test)
    logger.info('Max tokens train: {}'.format(max_token_train))
    logger.info('Max tokens dev: {}'.format(max_token_dev))
    logger.info('Max tokens test: {}'.format(max_token_test))
    logger.info('Max tokens: {}'.format(max_token))

    X_train, Y_train, train_mask = padding_email_sequences(x_train, y_train, max_token, post_padding=True)
    X_dev, Y_dev, dev_mask = padding_email_sequences(x_dev, y_dev, max_token, post_padding=True)
    X_test, Y_test, test_mask = padding_email_sequences(x_test, y_test, max_token, post_padding=True)

    X_train_char = pad_sequences(x_train_char, maxlen=300, truncating='post')
    X_dev_char = pad_sequences(x_dev_char, maxlen=300, truncating='post')
    X_test_char = pad_sequences(x_test_char, maxlen=300, truncating='post')

    logger.info('X train shape: {}'.format(X_train.shape))
    logger.info('X dev shape: {}'.format(X_dev.shape))
    logger.info('X test shape: {}'.format(X_test.shape))

    logger.info('X train char shape: {}'.format(X_train_char.shape))
    logger.info('X dev char shape: {}'.format(X_dev_char.shape))
    logger.info('X test char shape: {}'.format(X_test_char.shape))

    logger.info('Y train shape: {}'.format(Y_train.shape))
    logger.info('Y dev shape: {}'.format(Y_dev.shape))
    logger.info('Y test shape: {}'.format(Y_test.shape))

    if embedding_path:
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    if embedd_matrix is not None:
        embedd_dim = embedd_matrix.shape[1]
        embed_table = [embedd_matrix]
    
    model = build_simple_themis_word_char(vocab, char_vocab, max_token, max_char_token, embedd_dim, embed_table)
    evaluator = Evaluator(model, X_train, X_dev, X_test, X_train_char, X_dev_char, X_test_char, Y_train, Y_dev, Y_test, batch_size)

    logger.info("Initial evaluation: ")
    evaluator.predict()
    evaluator.print_eval()

    logger.info("Train model")
    for ii in range(epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time()
        model.fit([X_train, X_train_char], Y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evaluator.predict()
        evaluator.print_eval()
    evaluator.print_final_eval()
    

if __name__ == "__main__":
    main()
