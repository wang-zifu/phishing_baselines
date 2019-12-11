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
    logger.info('Training')
    model.fit(X_train, Y_train,
            batch_size=32,
            epochs=20)

    logger.info('Evaluation')
    loss, acc = model.evaluate(X_test, Y_test,
                            batch_size=32)
    logger.info('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


if __name__ == "__main__":
    main()
