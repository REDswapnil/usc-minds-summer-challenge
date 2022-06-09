import json
import logging
import pathlib
import pickle
import sys
import nltk
from nltk import word_tokenize
from nltk.corpus import words

from sentiment_analysis import pre_process_clean


def do_predict(classifier):
    logger.info('Predicting sentiment of news articles..')
    with open('articles.json') as f:
        articles = json.loads(f.read())
        for article in articles:
            temp = article['content'].replace("&shy", '').replace(";\xad", "").replace("\xad", "")
            custom_tokens = pre_process_clean(word_tokenize(temp), USELESS_WORDS, VALID_ENGLISH_WORDS)
            logger.info(f'Article: {temp} \n Sentiment : {classifier.classify(dict([token, True] for token in custom_tokens))}\n')


if __name__ == "__main__":

    APP_LOGGER = 'logger'

    # Update NLTK corpus
    nltk.download([
        "stopwords",
        "movie_reviews",
        "punkt",
        "wordnet",
        "averaged_perceptron_tagger",
        "omw-1.4",
        "words"])

    # Initialize Logger
    logger = logging.getLogger(APP_LOGGER)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s -> %(message)s\n')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Get stop words from nltk corpus
    USELESS_WORDS = nltk.corpus.stopwords.words("english")

    # Set of valid english words
    VALID_ENGLISH_WORDS = set(words.words())

    classifier = None
    logger.info('Loading model from pickle file...')
    if pathlib.Path('articles.json').exists() and pathlib.Path('model.pickle').exists():
        with open('model.pickle', 'rb') as f:
            classifier = pickle.load(f)
    else:
        logger.error("Either articles.json or model.pickle not found. Exiting...")
        sys.exit(1)

    do_predict(classifier)

