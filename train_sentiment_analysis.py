import pandas as pd
import re
import numpy as np
import argparse
from time import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models.themis_models import build_simple_themis
from evaluators.evaluator_sentiment_word_only import Evaluator
from utils.clean_data import process_legit_phish_data
from utils.preprocess_data import split_data, extract_labels, create_vocab, read_dataset
from utils.general_utils import get_logger, padding_email_sequences, load_word_embedding_dict, build_embedd_table


TAG_RE = re.compile(r'<[^>]+>')
logger = get_logger("Train sent class...")


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    sentence = remove_tags(sen)
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def main():
    parser = argparse.ArgumentParser(description="Word themis model")
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=16, help='Number of emails in each batch')
    parser.add_argument('--embedding', type=str, default='glove', help='Word embedding type, word2vec, senna or glove')
    parser.add_argument('--embedding_dim', type=int, default=50, help='Dimension of embedding')
    parser.add_argument('--embedding_path', type=str, default='embeddings/glove.6B.50d.txt', help='Path to embedding vec file')
    parser.add_argument('--seed', type=int, default=42, help='Set seed for data split')
    parser.add_argument('--legit_path', type=str, default='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/legit/',
                        help='Path to legit emails folder')
    parser.add_argument('--phish_path', type=str, default='ISWPA2.0 Train Data/IWSPA2.0_Training_No_Header/phish/',
                        help='Path to phish emails folder')
    parser.add_argument('--baby', action='store_true', help='Set to True for small data quantity for debug')

    args = parser.parse_args()
    epochs = args.num_epochs
    batch_size = args.batch_size
    embedding_path = args.embedding_path
    embedding = args.embedding
    embedd_dim = args.embedding_dim
    legit_path = args.legit_path
    phish_path = args.phish_path
    seed = args.seed
    baby = args.baby

    all_data = process_legit_phish_data(legit_path=legit_path, phish_path=phish_path)

    train, dev, test = split_data(all_data, random_state=seed)
    x_train_phish, y_train_phish = extract_labels(train)
    vocab = create_vocab(x_train_phish, vocab_size=20000, to_lower=True)

    data_path = 'ISWPA2.0 Train Data/IMDB Dataset.csv'
    if baby:
        movie_reviews = pd.read_csv(data_path)[:100]
    else:
        movie_reviews = pd.read_csv(data_path)
    movie_reviews.isnull().values.any()
    X = []
    sentences = list(movie_reviews['review'])
    for sen in sentences:
        X.append(preprocess_text(sen))

    X, max_token = read_dataset(X, vocab, to_lower=True)
    max_token = 300

    y = movie_reviews['sentiment']
    y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
    X_train, X_test_dev, y_train, y_test_dev = train_test_split(X, y, test_size=0.40, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test_dev, y_test_dev, test_size=0.50, random_state=42)

    X_train = pad_sequences(X_train, maxlen=300, truncating='post')
    X_dev = pad_sequences(X_dev, maxlen=300, truncating='post')
    X_test = pad_sequences(X_test, maxlen=300, truncating='post')

    logger.info('X train shape: {}'.format(X_train.shape))
    logger.info('X dev shape: {}'.format(X_dev.shape))
    logger.info('X test shape: {}'.format(X_test.shape))

    logger.info('Y train shape: {}'.format(y_train.shape))
    logger.info('Y dev shape: {}'.format(y_dev.shape))
    logger.info('Y test shape: {}'.format(y_test.shape))

    if embedding_path:
        embedd_dict, embedd_dim, _ = load_word_embedding_dict(embedding, embedding_path, vocab, logger, embedd_dim)
        embedd_matrix = build_embedd_table(vocab, embedd_dict, embedd_dim, logger, caseless=True)
    else:
        embedd_matrix = None

    if embedd_matrix is not None:
        embedd_dim = embedd_matrix.shape[1]
        embed_table = [embedd_matrix]

    model = build_simple_themis(vocab, max_token, embedd_dim, embed_table)

    save_path = 'saved_models/word_only_themis_seed' + str(seed)
    evaluator = Evaluator(model, X_train, X_dev, X_test, y_train, y_dev, y_test, batch_size, save_path)

    logger.info("Initial evaluation: ")
    evaluator.predict()
    evaluator.print_eval()

    logger.info("Train model")
    for ii in range(epochs):
        logger.info('Epoch %s/%s' % (str(ii + 1), epochs))
        start_time = time()
        model.fit(X_train, y_train, batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        tt_time = time() - start_time
        logger.info("Training one epoch in %.3f s" % tt_time)
        evaluator.predict()
        evaluator.print_eval()

    evaluator.print_final_eval()


if __name__ == "__main__":
    main()