import json
import logging
from dataclasses import asdict
from typing import List
from dto.article import Article
from exception.app_exception import AppException
from util.request_handler import RequestHandler
from util.article_parser import ArticleParser


def get_articles():
    requestor = RequestHandler()
    article_parser = ArticleParser()
    if not RESOURCE_BASE_URL or not TARGET_RESOURCE_PATH:
        logger.error("Cannot find resource URL")
        raise AppException("Cannot find resource URL")
    logger.info(f'Requesting resource URL - {RESOURCE_BASE_URL + TARGET_RESOURCE_PATH}')
    resp = requestor.get(RESOURCE_BASE_URL + TARGET_RESOURCE_PATH)
    try:
        article_ref_list: List[Article] = article_parser.get_recent_article_references(resp.text)
    except Exception as e:
        logger.error('Exception occurred while obtaining articles from resource URL', e)
    # for i in range(len(article_ref_list)):
    #     resp = requestor.get(RESOURCE_BASE_URL + article_ref_list[i].reference)
    #     content = article_parser.get_article_content(resp.text)
    #     article_ref_list[i].content = content
    else:
        write_json_article_file(article_ref_list)


def write_json_article_file(articles: List[Article]) -> None:
    logger.info("Writing articles JSON file...")
    article_list: List = list()
    for article in articles:
        article_list.append(asdict(article))
    with open('articles.json', 'w') as a:
        a.write(json.dumps(article_list))


if __name__ == "__main__":
    RESOURCE_BASE_URL = 'https://www.aljazeera.com/'
    TARGET_RESOURCE_PATH = 'where/mozambique/'
    APP_LOGGER = 'logger'

    # Initialize Logger
    logger = logging.getLogger(APP_LOGGER)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s -> %(message)s\n')
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # Run app
    logger.info('Running app')
    get_articles()
