import weakref as wr

from juxtorpus.viz.corpus import wordcloud, timeline, wordclouds, timelines


class CorpusViz(object):
    """ This is a container visualisation class that is a part of corpus. """

    def __init__(self, corpus):
        self._corpus = wr.ref(corpus)

    def wordcloud(self, metric: str = 'tf', max_words: int = 50, word_type: str = 'word', stopwords: list[str] = None):
        return wordcloud(self._corpus(), metric=metric, max_words=max_words, word_type=word_type, stopwords=stopwords)

    def timeline(self, datetime_meta: str, freq: str):
        return timeline(self._corpus(), datetime_meta=datetime_meta, freq=freq)


class CorporaViz(object):
    def __init__(self, corpora):
        self._corpora = wr.ref(corpora)

    def wordclouds(self, names: list[str], metric: str = 'tf', max_words: int = 50, word_type: str = 'word', stopwords: list[str] = None ):
        return wordclouds(corpora=self._corpora(), names=names,
                          metric=metric, max_words=max_words, word_type=word_type, stopwords=stopwords)

    def timelines(self, names: list[str], datetime_meta: str, freq: str):
        return timelines(corpora=self._corpora(), names=names,
                         datetime_meta=datetime_meta, freq=freq)
