import os
import glob
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from bs4 import BeautifulSoup
import re
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords
from nltk.stem.porter import *

from wordcloud import WordCloud, STOPWORDS

def plot_wordcloud(data, sentiment):
    combined_text = " ".join([review for review in data['train'][sentiment]])

    wc = WordCloud(background_color='white', max_words=50,
                   stopwords=STOPWORDS.update(['br', 'film', 'movie']))

    plt.imshow(wc.generate(combined_text))
    plt.axis('off')
    plt.show()


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

def prepare_data(data):
    """
    Prepare training and test sets from IMDb movie reviews.
    """
    train_pos_reviews_list = data['train']['pos']
    train_neg_reviews_list = data['train']['neg']
    test_pos_reviews_list = data['test']['pos']
    test_neg_reviews_list = data['test']['neg']

    train_pos_label_list = len(train_pos_reviews_list) * ['pos']
    train_neg_label_list = len(train_neg_reviews_list) * ['neg']
    test_pos_label_list = len(test_pos_reviews_list) * ['pos']
    test_neg_label_list = len(test_neg_reviews_list) * ['neg']

    train_reviews_list = train_pos_reviews_list + train_neg_reviews_list
    train_labels_list = train_pos_label_list + train_neg_label_list
    test_reviews_list = test_pos_reviews_list + test_neg_reviews_list
    test_labels_list = test_pos_label_list + test_neg_label_list

    data_train, labels_train = shuffle(train_reviews_list, train_labels_list)
    data_test, labels_test = shuffle(test_reviews_list, test_labels_list)

    # Return a unified training data, test data, training labels, test labels
    return data_train, data_test, labels_train, labels_test


def review_to_words(review):
    """Convert a raw review string into a sequence of words."""
    stemmer = PorterStemmer()

    # remove html tags
    soup = BeautifulSoup(review, "html5lib")
    text = soup.get_text()

    # convert to lowercase
    text = text.lower()

    # remove non-letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # tokenize
    words = text.split()

    # remove stopwords
    words = [w for w in words if w not in stopwords.words("english")]

    # stemming
    stemmed = [stemmer.stem(w) for w in words]

    return stemmed

def preprocess_data():
    raise NotImplementedError

def extract_bow_features():
    raise NotImplementedError

def classify_gboost():
    raise NotImplementedError

def main():
    data, labels = read_imdb_data('data')
    print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
        len(data['train']['pos']), len(data['train']['neg']),
        len(data['test']['pos']), len(data['test']['neg'])))
    plot_wordcloud(data, 'neg')
    plot_wordcloud(data, 'pos')


if __name__ == "__main__":
    main()
