from sklearn.model_selection import train_test_split
from utils.general_utils import get_logger

logger = get_logger("Pre-process data ...")


def split_data(data, random_state=42):
    train, dev_test = train_test_split(
        data, test_size=0.4, random_state=random_state)
    dev, test = train_test_split(
        dev_test, test_size=0.5, random_state=random_state)
    return train, dev, test


def extract_labels(data):
    x_data = []
    y_data = []
    for item in data:
        x_data.append(item[0])
        y_data.append(item[1])
    return x_data, y_data


def create_vocab(data, vocab_size, to_lower):
    logger.info('Creating vocabulary')
    total_words, unique_words = 0, 0
    word_freqs = {}
    for content in data:
        if to_lower:
            content = [w.lower() for w in content]
        for word in content:
            try:
                word_freqs[word] += 1
            except KeyError:
                unique_words += 1
                word_freqs[word] = 1
            total_words += 1
    logger.info('  %i total words, %i unique words' % (total_words, unique_words))
    import operator
    sorted_word_freqs = sorted(word_freqs.items(), key=operator.itemgetter(1), reverse=True)
    if vocab_size <= 0:
        vocab_size = 0
        for word, freq in sorted_word_freqs:
            if freq > 1:
                vocab_size += 1
    vocab = {'<pad>': 0, '<unk>': 1}
    vcb_len = len(vocab)
    index = vcb_len
    for word, _ in sorted_word_freqs[:vocab_size - vcb_len]:
        vocab[word] = index
        index += 1
    return vocab


def create_char_vocab(data, to_lower):
    logger.info("Create char vocabulary")
    total_chars, unique_chars = 0, 0
    char_vocab = {}
    start_index = 1
    char_vocab['<unk>'] = start_index
    next_index = start_index + 1
    for content in data:
        if to_lower:
            content = [w.lower() for w in content]
        for word in content:
            for char in list(word):
                if not char in char_vocab:
                    char_vocab[char] = next_index
                    next_index += 1
                    unique_chars += 1
                total_chars += 1
    logger.info('  %i total chars, %i unique chars' % (total_chars, unique_chars))
    return char_vocab


def read_dataset(data, vocab, to_lower):
    logger.info('Reading dataset just words')
    data_x = []
    unk_hit, total, long_count = 0., 0., 0.
    max_tokennum = -1
    for content in data:
        if to_lower:
            tokens = [w.lower() for w in content]
        else:
            tokens = content
        indices = []
        length = len(tokens)
        if max_tokennum < length:
            max_tokennum = length
        if length > 2000:
            long_count += 1
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1
        data_x.append(indices)
    logger.info('  <unk> hit rate: %.2f%%' % (100*unk_hit/total))
    logger.info(' number of long emails: {}'.format(long_count))
    return data_x, max_tokennum


def read_dataset_word_char(data, vocab, char_vocab, to_lower):
    logger.info('Reading dataset words and chars')
    data_x, char_x = [], []
    unk_hit, total, long_count = 0., 0., 0.
    max_tokennum = -1
    max_charnum = -1
    for content in data:
        if to_lower:
            tokens = [w.lower() for w in content]
        else:
            tokens = content
        indices = []
        c_indices = []
        length = len(tokens)
        if max_tokennum < length:
            max_tokennum = length
        if length > 2000:
            long_count += 1
        for token in tokens:
            if token in vocab:
                indices.append(vocab[token])
            else:
                indices.append(vocab['<unk>'])
                unk_hit += 1
            total += 1

            # Characters per token
            current_chars = list(token)
            for c in current_chars:
                try:
                    c_indices.append(char_vocab[c])
                except:
                    c_indices.append(char_vocab['<unk>'])
            if len(c_indices) > max_charnum:
                max_charnum = len(c_indices)
        data_x.append(indices)
        char_x.append(c_indices)
    logger.info('  <unk> hit rate: %.2f%%' % (100*unk_hit/total))
    logger.info(' number of long emails: {}'.format(long_count))
    return data_x, char_x, max_tokennum, max_charnum