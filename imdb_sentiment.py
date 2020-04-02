import os
import glob

def read_imdb_data(data_dir):
    """
    Read review data from specified directory and return reviews and labels(pos/neg) as nested dictionaries
    :param data_dir: the directory where data reside.
                     Assumed directory structure:
                     - data/
                         - train/
                             - pos/
                             - neg/
                         - test/
                             - pos/
                             - neg/
    :return: nested dictionaries containing data and reviews
    """
    data = {}
    labels = {}

    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}

        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []

            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)

            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    labels[data_type][sentiment].append(sentiment)
    return data, labels

def prepare_data():
    raise NotImplementedError

def review_word_tokenizer():
    raise NotImplementedError

def preprocess_data():
    raise NotImplementedError

def extract_bow_features():
    raise NotImplementedError

def classify_gboost():
    raise NotImplementedError

def main():
    data, labels = read_imdb_data('data')

if __name__ == "__main__":
    main()
