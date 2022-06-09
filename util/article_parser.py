import logging

from typing import List
from bs4 import BeautifulSoup
from dto.article import Article


logger = logging.getLogger('app_logger')


class ArticleParser:

    FEATURED_ARTICLE_CLASS_2 = "featured-articles-list__item"
    FEATURED_ARTICLE_CLASS = ["u-clickable-card", "gc--type-post"]
    ANCHOR_TAG_CLASS = "u-clickable-card__link"
    ARTICLE_CONTENT_DIV_CLASS = ["wysiwyg", "wysiwyg--all-content"]
    ARTICLE_LIMIT = 10

    def get_recent_article_references(self, resource_html: str) -> List[Article]:
        logger.info('Getting recent articles..')
        article_list: List[Article] = list()
        soup = BeautifulSoup(resource_html, 'html.parser')
        for index, article_ele in enumerate(soup.find_all('article', {'class': self.FEATURED_ARTICLE_CLASS})):
            if index >= self.ARTICLE_LIMIT:
                break
            a_tag = article_ele.find('a', {'class': self.ANCHOR_TAG_CLASS})
            title = str(a_tag.next.string).strip()
            href = a_tag.attrs.get('href')
            content = str(article_ele.prettify(
    formatter=lambda x: x.replace(u'\xad', '')).contents[1].contents[1].string)
            article_list.append(Article(title=title, reference=href, content=content))
        return article_list

    def get_article_content(self, resource_html: str) -> str:
        logger.info('Getting article content...')
        content = str()
        soup = BeautifulSoup(resource_html, 'html.parser')
        div_tag = soup.find('div', {'class': self.ARTICLE_CONTENT_DIV_CLASS})
        p_tags = div_tag.find_all('p')
        for p_tag in p_tags:
            content = content + str(p_tag.next).strip()
        return content




