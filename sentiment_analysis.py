import json
import logging
import random
import string
import nltk
import pandas as pd
import sys
import pathlib

from nltk import classify, NaiveBayesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import words


def pre_process_clean(tokens, useless_words, valid_english_word_set):
    cleaned_tokens = list()
    lemma = WordNetLemmatizer()

    for token, tag in pos_tag(tokens):
        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        token = lemma.lemmatize(token, pos)
        token = token.replace("&shy", "").replace(";\xad", "").replace("\xad", "")

        if token in valid_english_word_set and \
                len(token) > 0 and \
                token not in string.punctuation and \
                token.lower() not in useless_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens


def get_tokens_generator(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tokens)


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

    if len(sys.argv) == 2 and sys.argv[1] == 'articles.json':
        if not pathlib.Path(sys.argv[1]).exists():
            logger.error(f'{sys.argv[1]} not found in the project dir. To skip processing of {sys.argv[1]}, '
                         f'use: python sentiment_analysis.py. Exiting..')
            sys.exit(1)
        else:
            logger.info('Will predict articles in articles.json at the end..')
    else:
        logger.info('Will skip processing of articles.json via classifier at the end...')

    # Set of valid english words
    VALID_ENGLISH_WORDS = set(words.words())

    # Get stop words from nltk corpus
    USELESS_WORDS = nltk.corpus.stopwords.words("english")

    # Read IMDB dataset
    logger.info('Reading IMDB dataset into Pandas Dataframe')
    imdb_data = pd.read_csv('IMDB_Dataset.csv')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    # Get separate positive and negative reviews
    logger.info('Getting separate positive and negative reviews')
    pos = imdb_data[imdb_data['sentiment'] == 'positive']
    neg = imdb_data[imdb_data['sentiment'] == 'negative']

    # Tokenize and clean tokens
    logger.info('Tokenizing and cleaning tokens, for positive and negative reviews')
    for pos_reviews in tqdm(pos.review):
        positive_cleaned_tokens_list.append(pre_process_clean(word_tokenize(pos_reviews),
                                                              USELESS_WORDS, VALID_ENGLISH_WORDS))

    for neg_reviews in tqdm(neg.review):
        negative_cleaned_tokens_list.append(pre_process_clean(word_tokenize(neg_reviews),
                                                              USELESS_WORDS, VALID_ENGLISH_WORDS))

    positive_tokens_for_model = get_tokens_generator(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tokens_generator(negative_cleaned_tokens_list)

    # Creating dataset as per NLTK Naive Bayes classifier
    logger.info('Creating dataset as per NLTK Naive Bayes classifier')
    positive_dataset = [(review_dict, "Positive")
                        for review_dict in positive_tokens_for_model]

    negative_dataset = [(review_dict, "Negative")
                        for review_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    # Shuffling
    random.shuffle(dataset)

    # Train - Test Split
    train_data = dataset[:40000]
    test_data = dataset[40000:]

    # Train
    logger.info('Starting to train Naive Bayes Classifier')
    train_data_with_progress_bar = tqdm(train_data)
    classifier = NaiveBayesClassifier.train(train_data_with_progress_bar)

    logger.info(f"Accuracy is: {classify.accuracy(classifier, test_data) * 100}%")

    classifier.show_most_informative_features(10)

    # Run classifier on articles.json
    if len(sys.argv) == 2 and sys.argv[1] == 'articles.json':
        with open('articles.json') as f:
            articles = json.loads(f.read())
            for article in articles:
                temp = article['content'].replace("&shy", '').replace(";\xad", "").replace("\xad", "")
                custom_tokens = pre_process_clean(word_tokenize(temp), USELESS_WORDS, VALID_ENGLISH_WORDS)
                logger.info(f'Article: {temp} \n Sentiment : {classifier.classify(dict([token, True] for token in custom_tokens))}\n')

    logger.info('End of script !!')
