from time import time
import keras.backend as K
import numpy as np
from keras.models import model_from_json
import argparse
from utils.clean_data import process_legit_phish_data
from utils.preprocess_data import split_data, extract_labels, create_vocab, read_dataset, create_char_vocab
from utils.general_utils import get_logger, padding_email_sequences, load_word_embedding_dict, build_embedd_table
from models.themis_models import build_simple_themis
from evaluators.evaluator_word_only import Evaluator

logger = get_logger("Train ...")


def main():
    parser = argparse.ArgumentParser(description="Word and char themis model")
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of emails in each batch')
    parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Dimension of embedding')
    parser.add_argument('--embedding_path', type=str, default='embeddings/glove.6B.50d.txt',
                        help='Path to embedding vec file')
    parser.add_argument('--baby', action='store_true', help='Set to True for small data quantity for debug')
    parser.add_argument('--seed', type=int, default=42, help='Set seed for data split')
    parser.add_argument('--legit_path', type=str, default='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/legit/',
                        help='Path to legit emails folder')
    parser.add_argument('--phish_path', type=str, default='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/phish/',
                        help='Path to phish emails folder')

    args = parser.parse_args()
    legit_path = args.legit_path
    phish_path = args.phish_path
    embedding_path = args.embedding_path
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    epochs = args.num_epochs
    batch_size = args.batch_size
    baby = args.baby
    seed = args.seed
    all_data = process_legit_phish_data(legit_path=legit_path, phish_path=phish_path)

    if baby:
        all_data = all_data[:100]
    train, dev, test = split_data(all_data)
    x_train, y_train = extract_labels(train)
    x_dev, y_dev = extract_labels(dev)
    x_test, y_test = extract_labels(test)
    vocab = create_vocab(x_train, vocab_size=20000, to_lower=True)

    x_train, max_token_train = read_dataset(x_train, vocab, to_lower=True)
    x_dev, max_token_dev = read_dataset(x_dev, vocab, to_lower=True)
    x_test, max_token_test = read_dataset(x_test, vocab, to_lower=True)

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

    json_file = open('model.json', 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    model.load_weights('model.h5')
    evaluator = Evaluator(model, X_train, X_dev, X_test, Y_train, Y_dev, Y_test, batch_size)

    logger.info("Initial evaluation: ")
    evaluator.predict()
    evaluator.print_eval()

    logger.info("Train model")
    for ii in range(epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time()
        model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evaluator.predict()
        evaluator.print_eval()

    evaluator.print_final_eval()


if __name__ == "__main__":
    main()
