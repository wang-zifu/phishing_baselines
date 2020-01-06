import pandas as pd
import re
import numpy as np
import argparse
from time import time
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from models.themis_models import build_simple_themis
from evaluators.evaluator_word_only import Evaluator
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

    args = parser.parse_args()
    epochs = args.num_epochs
    batch_size = args.batch_size
    embedding_path = args.embedding_path
    embedding = args.embedding
    embedd_dim = args.embedding_dim

    data_path = 'ISWPA2.0 Train Data/IMDB Dataset.csv'
    movie_reviews = pd.read_csv(data_path)
    movie_reviews.isnull().values.any()
    X = []
    sentences = list(movie_reviews['review'])
    for sen in sentences:
        X.append(preprocess_text(sen))
    
    y = movie_reviews['sentiment']
    y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))
    X_train, X_test_dev, y_train, y_test_dev = train_test_split(X, y, test_size=0.40, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(X_test_dev, y_test_dev, test_size=0.50, random_state=42)

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(X_train)

    X_train = tokenizer.texts_to_sequences(X_train)
    X_dev = tokenizer.texts_to_sequences(X_dev)
    X_test = tokenizer.texts_to_sequences(X_test)

    vocab_size = len(tokenizer.word_index) + 1
    maxlen = 300
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_dev = pad_sequences(X_dev, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

    embeddings_dictionary = dict()
    glove_file = open(embedding_path)
    for line in glove_file:
        records = line.split()
        word = records[0]
        vector_dimensions = np.asarray(records[1:], dtype='float32')
        embeddings_dictionary [word] = vector_dimensions
    glove_file.close()

    embedding_matrix = np.zeros((vocab_size, 50))
    for word, index in tokenizer.word_index.items():
        embedding_vector = embeddings_dictionary.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    embedding_matrix = [embedding_matrix]
    
    model = build_simple_themis(vocab_size, maxlen, embedd_dim, embedding_matrix)

    evaluator = Evaluator(model, X_train, X_dev, X_test, y_train, y_dev, y_test, batch_size)

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