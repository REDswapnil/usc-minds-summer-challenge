import requests


class RequestHandler:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:98.0) Gecko/20100101 '
                                                   'Firefox/98.0'})
        self.timeout = 60

    def request(self, method, url, **kwargs):
        return self.session.request(method, url, **kwargs)

    def head(self, url, **kwargs):
        return self.session.head(url, **kwargs)

    def get(self, url, **kwargs):
        resp = self.session.get(url, timeout=self.timeout, **kwargs)
        resp.raise_for_status()
        return resp

    def post(self, url, **kwargs):
        return self.session.post(url, **kwargs)

    def put(self, url, **kwargs):
        return self.session.put(url, **kwargs)

    def patch(self, url, **kwargs):
        return self.session.patch(url, **kwargs)

    def delete(self, url, **kwargs):
        return self.session.delete(url, **kwargs)
