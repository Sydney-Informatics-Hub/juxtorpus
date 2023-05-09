import weakref as wr

from juxtorpus.viz.corpus import wordcloud, timeline


class CorpusViz(object):
    """ This is a container visualisation class that is a part of corpus. """

    def __init__(self, corpus):
        self._corpus = wr.ref(corpus)

    def wordcloud(self, metric: str = 'tf', max_words: int = 50, word_type: str = 'word'):
        return wordcloud(self._corpus(), metric=metric, max_words=max_words, word_type=word_type)

    def timeline(self, datetime_meta: str, freq: str):
        return timeline(self._corpus(), datetime_meta=datetime_meta, freq=freq)
