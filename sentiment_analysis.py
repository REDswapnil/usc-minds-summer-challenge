import json
import logging
import random
import string
import nltk
import pandas as pd
import time

from nltk import classify, NaiveBayesClassifier
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from nltk.corpus import words


def pre_process_clean(tokens, useless_words):
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
        token = token.replace("&shy", '').replace(";\xad", "").replace("\xad", "")
        if token not in VALID_ENGLISH_WORDS:
            continue

        if len(token) > 0 and token not in string.punctuation and token.lower() not in useless_words:
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

    VALID_ENGLISH_WORDS = set(words.words())

    # Initialize Logger
    logger = logging.getLogger(APP_LOGGER)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s -> %(message)s\n')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Get stop words from nltk corpus
    useless_words = nltk.corpus.stopwords.words("english")

    imdb_data = pd.read_csv('IMDB_Dataset.csv')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []

    pos = imdb_data[imdb_data['sentiment'] == 'positive']
    neg = imdb_data[imdb_data['sentiment'] == 'negative']

    for pos_reviews in tqdm(pos.review):
        positive_cleaned_tokens_list.append(pre_process_clean(word_tokenize(pos_reviews), useless_words))

    for neg_reviews in tqdm(neg.review):
        negative_cleaned_tokens_list.append(pre_process_clean(word_tokenize(neg_reviews), useless_words))

    positive_tokens_for_model = get_tokens_generator(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tokens_generator(negative_cleaned_tokens_list)

    positive_dataset = [(review_dict, "Positive")
                        for review_dict in positive_tokens_for_model]

    negative_dataset = [(review_dict, "Negative")
                        for review_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    random.shuffle(dataset)

    train_data = dataset[:40000]
    test_data = dataset[40000:]

    train_data_with_progress_bar = tqdm(train_data)
    classifier = NaiveBayesClassifier.train(train_data_with_progress_bar)

    print(f"Accuracy is: {classify.accuracy(classifier, test_data) * 100}%")

    print(classifier.show_most_informative_features(10))

    with open('articles.json') as f:
        articles = json.loads(f.read())
        for article in articles:
            temp = article['content'].replace("&shy", '').replace(";\xad", "").replace("\xad", "")
            custom_tokens = pre_process_clean(word_tokenize(temp), useless_words)
            print(temp + '\n', classifier.classify(dict([token, True] for token in custom_tokens)), '\n')
