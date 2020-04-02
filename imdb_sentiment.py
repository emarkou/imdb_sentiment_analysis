import os
import glob
import pickle
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import sklearn.preprocessing as pr
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier

from bs4 import BeautifulSoup
import re
import nltk
nltk.download("stopwords")   # download list of stopwords (only once; need not run it again)
from nltk.corpus import stopwords
from nltk.stem.porter import *

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

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
    soup = BeautifulSoup(review, "html.parser")
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


def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=os.path.join("cache", "sentiment_analysis"), cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        words_train = list(map(review_to_words, data_train))
        words_test = list(map(review_to_words, data_test))

        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                                                              cache_data['words_test'], cache_data['labels_train'],
                                                              cache_data['labels_test'])

    return words_train, words_test, labels_train, labels_test


def extract_bow_features(words_train, words_test, vocabulary_size=5000,
                         cache_dir=os.path.join("cache", "sentiment_analysis"), cache_file="bow_features.pkl"):
    """Extract Bag-of-Words for a given set of documents, already preprocessed into words."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = joblib.load(f)
            print("Read features from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay

    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # NOTE: Training documents have already been preprocessed and tokenized into words;
        #       pass in dummy functions to skip those steps, e.g. preprocessor=lambda x: x
        vectorizer = CountVectorizer(max_features=vocabulary_size, preprocessor=lambda x: x, tokenizer=lambda x: x)
        features_train = vectorizer.fit_transform(words_train)
        features_train = features_train.toarray()

        features_test = vectorizer.fit_transform(words_test)
        features_test = features_test.toarray()

        # Write to cache file for future runs (store vocabulary as well)
        if cache_file is not None:
            vocabulary = vectorizer.vocabulary_
            cache_data = dict(features_train=features_train, features_test=features_test,
                              vocabulary=vocabulary)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                joblib.dump(cache_data, f)
            print("Wrote features to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        features_train, features_test, vocabulary = (cache_data['features_train'],
                                                     cache_data['features_test'], cache_data['vocabulary'])

    # Return both the extracted features as well as the vocabulary
    return features_train, features_test, vocabulary

def normalize_vector(data):
    return pr.normalize(data, axis=1)

def classify_gaussian_nb(features_train, features_test, labels_train, labels_test):
    clf = GaussianNB()
    clf.fit(features_train, labels_train)

    # Calculate the mean accuracy score on training and test sets
    print("[{}] Accuracy: train = {}, test = {}".format(
        clf.__class__.__name__,
        clf.score(features_train, labels_train),
        clf.score(features_test, labels_test)))
    return clf


def classify_gboost(features_train, features_test, labels_train, labels_test, n_estimators=100):

    clf = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=1.0, max_depth=1, random_state=0)
    clf.fit(features_train, labels_train)
    print("[{}] Accuracy: train = {}, test = {}".format(
        clf.__class__.__name__,
        clf.score(features_train, labels_train),
        clf.score(features_test, labels_test)))
    return clf

def main():
    data, labels = read_imdb_data('data')

    print("IMDb reviews: train = {} pos / {} neg, test = {} pos / {} neg".format(
        len(data['train']['pos']), len(data['train']['neg']),
        len(data['test']['pos']), len(data['test']['neg'])))
    #plot_wordcloud(data, 'neg')
    #plot_wordcloud(data, 'pos')

    cache_dir = os.path.join("cache", "sentiment_analysis")
    os.makedirs(cache_dir, exist_ok=True)

    data_train, data_test, labels_train, labels_test = prepare_data(data)
    words_train, words_test, labels_train, labels_test = preprocess_data(
        data_train, data_test, labels_train, labels_test)

    features_train, features_test, vocabulary = extract_bow_features(words_train, words_test)
    features_train_norm = normalize_vector(features_train)
    features_test_norm = normalize_vector(features_test)

    gaussian_clf = classify_gaussian_nb(features_train_norm, features_test_norm, labels_train, labels_test)
    gboost_clf = classify_gboost(features_train_norm, features_test_norm, labels_train, labels_test)

    joblib.dump(gaussian_clf, 'imdb_gaussian_clf.joblib')
    joblib.dump(gboost_clf, 'imdb_gaussian_clf.joblib')


if __name__ == "__main__":
    main()
