from time import time
import keras.backend as K
import numpy as np
from sklearn.metrics import accuracy_score
from utils.metrics import precision, recall, f1_score, false_positive_rate
from utils.clean_data import process_legit_phish_data
from utils.preprocess_data import split_data, extract_labels, create_vocab, read_dataset
from utils.general_utils import get_logger, padding_email_sequences, load_word_embedding_dict, build_embedd_table
from models.themis_models import build_simple_themis

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
    vocab = create_vocab(x_train, vocab_size=20000, to_lower=False)
    x_train, max_token_train = read_dataset(x_train, vocab, to_lower=False)
    x_dev, max_token_dev = read_dataset(x_dev, vocab, to_lower=False)
    x_test, max_token_test = read_dataset(x_test, vocab, to_lower=False)

    max_token = max(max_token_train, max_token_dev, max_token_test)
    logger.info('Max tokens train: {}'.format(max_token_train))
    logger.info('Max tokens dev: {}'.format(max_token_dev))
    logger.info('Max tokens test: {}'.format(max_token_test))
    logger.info('Max tokens: {}'.format(max_token))

    X_train, Y_train, train_mask = padding_email_sequences(x_train, y_train, max_token, post_padding=True)
    X_dev, Y_dev, dev_mask = padding_email_sequences(x_dev, y_dev, max_token, post_padding=True)
    X_test, Y_test, test_mask = padding_email_sequences(x_test, y_test, max_token, post_padding=True)

    logger.info('X train shape: {}'.format(X_train.shape))
    logger.info('X dev shape: {}'.format(X_dev.shape))
    logger.info('X test shape: {}'.format(X_test.shape))

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
    
    model = build_simple_themis(vocab, max_token, embedd_dim, embed_table)

    logger.info("Initial evaluation: ")
    best_dev_acc = -1
    best_dev_precision = -1
    best_dev_recall = -1
    best_dev_f1 = -1
    best_dev_false_pos_rate = -1

    best_test_acc = -1
    best_test_precision = -1
    best_test_recall = -1
    best_test_f1 = -1
    best_test_false_pos_rate = -1
    
    train_pred = model.predict(X_train, batch_size=batch_size)
    dev_pred = model.predict(X_dev, batch_size=batch_size)
    test_pred = model.predict(X_test, batch_size=batch_size)
    
    train_pred = np.round(train_pred)
    dev_pred = np.round(dev_pred)
    test_pred = np.round(test_pred)

    dev_acc = accuracy_score(Y_dev, dev_pred)
    dev_precision = precision(Y_dev, dev_pred)
    dev_recall = recall(Y_dev, dev_pred)
    dev_f1 = f1_score(Y_dev, dev_pred)
    dev_false_pos_rate = f1_score(Y_dev, dev_pred)

    test_acc = accuracy_score(Y_test, test_pred)
    test_precision = precision(Y_test, test_pred)
    test_recall = recall(Y_test, test_pred)
    test_f1 = f1_score(Y_test, test_pred)
    test_false_pos_rate = f1_score(Y_test, test_pred)

    if dev_acc > best_dev_acc:
        best_dev_acc = dev_acc
        best_dev_precision = dev_precision
        best_dev_recall = dev_recall
        best_dev_f1 = dev_f1
        best_dev_false_pos_rate = dev_false_pos_rate

        best_test_acc = test_acc
        best_test_precision = test_precision
        best_test_recall = test_recall
        best_test_f1 = test_f1
        best_test_false_pos_rate = test_false_pos_rate

    logger.info('[DEV]   ACC:  %.3f, PREC: %.3f, REC: %.3f, F1: %.3f, FPR: %.3f \n(Best ACC: {{%.3f}}, Best PREC: {{%.3f}}, Best REC: {{%.3f}}, Best F1: {{%.3f}}, Best FPR: {{%.3f}})' % (
        dev_acc, dev_precision, dev_recall, dev_f1, dev_false_pos_rate, 
        best_dev_acc, best_dev_precision, best_dev_recall, best_dev_f1, best_dev_false_pos_rate))
    logger.info('[TEST]   ACC:  %.3f, PREC: %.3f, REC: %.3f, F1: %.3f, FPR: %.3f \n(Best ACC: {{%.3f}}, Best PREC: {{%.3f}}, Best REC: {{%.3f}}, Best F1: {{%.3f}}, Best FPR: {{%.3f}})' % (
        test_acc, test_precision, test_recall, test_f1, test_false_pos_rate, 
        best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_false_pos_rate))
    logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

    logger.info("Train model")
    for ii in range(epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time()
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        train_pred = model.predict(X_train, batch_size=batch_size)
        dev_pred = model.predict(X_dev, batch_size=batch_size)
        test_pred = model.predict(X_test, batch_size=batch_size)

        train_pred = np.round(train_pred)
        dev_pred = np.round(dev_pred)
        test_pred = np.round(test_pred)

        dev_acc = accuracy_score(Y_dev, dev_pred)
        dev_precision = precision(Y_dev, dev_pred)
        dev_recall = recall(Y_dev, dev_pred)
        dev_f1 = f1_score(Y_dev, dev_pred)
        dev_false_pos_rate = f1_score(Y_dev, dev_pred)

        test_acc = accuracy_score(Y_test, test_pred)
        test_precision = precision(Y_test, test_pred)
        test_recall = recall(Y_test, test_pred)
        test_f1 = f1_score(Y_test, test_pred)
        test_false_pos_rate = f1_score(Y_test, test_pred)

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_dev_precision = dev_precision
            best_dev_recall = dev_recall
            best_dev_f1 = dev_f1
            best_dev_false_pos_rate = dev_false_pos_rate

            best_test_acc = test_acc
            best_test_precision = test_precision
            best_test_recall = test_recall
            best_test_f1 = test_f1
            best_test_false_pos_rate = test_false_pos_rate
        
        logger.info('[DEV]   ACC:  %.3f, PREC: %.3f, REC: %.3f, F1: %.3f, FPR: %.3f \n(Best ACC: {{%.3f}}, Best PREC: {{%.3f}}, Best REC: {{%.3f}}, Best F1: {{%.3f}}, Best FPR: {{%.3f}})' % (
            dev_acc, dev_precision, dev_recall, dev_f1, dev_false_pos_rate, 
            best_dev_acc, best_dev_precision, best_dev_recall, best_dev_f1, best_dev_false_pos_rate))
        logger.info('[TEST]   ACC:  %.3f, PREC: %.3f, REC: %.3f, F1: %.3f, FPR: %.3f \n(Best ACC: {{%.3f}}, Best PREC: {{%.3f}}, Best REC: {{%.3f}}, Best F1: {{%.3f}}, Best FPR: {{%.3f}})' % (
            test_acc, test_precision, test_recall, test_f1, test_false_pos_rate, 
            best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_false_pos_rate))
        logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')

    logger.info(
            '--------------------------------------------------------------------------------------------------------------------------')
    logger.info('[DEV]   BEST ACC:  %.3f, BEST PREC: %.3f, BEST REC: %.3f, BEST F1: %.3f, BEST FPR: %.3f' % (
        best_dev_acc, best_dev_precision, best_dev_recall, best_dev_f1, best_dev_false_pos_rate))
    logger.info('[TEST]   BEST ACC:  %.3f, BEST PREC: %.3f, BEST REC: %.3f, BEST F1: %.3f, BEST FPR: %.3f' % (
        best_test_acc, best_test_precision, best_test_recall, best_test_f1, best_test_false_pos_rate))
    

if __name__ == "__main__":
    main()
