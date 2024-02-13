from atap_corpus import Corpus
# from juxtorpus.features.keywords import TFKeywords, TFIDFKeywords
from juxtorpus.features.polarity import Polarity
from juxtorpus.features.similarity import Similarity
from juxtorpus.stats import Statistics

import numpy as np


class Jux:
    """ Jux
    This is the main class for Juxtorpus. It takes in 2 corpus and exposes numerous functions
    to help contrast the two corpus.

    It is expected that the exposed functions are used as tools that serve as building blocks
    for your own further analysis.
    """

    def __init__(self, corpus_0: Corpus, corpus_1: Corpus):
        # NOTE: numeric variables are used to maintain consistency with column names in pandas
        if not isinstance(corpus_0, Corpus) or not isinstance(corpus_1, Corpus):
            raise ValueError("corpus_0 and corpus_1 must be a Corpus.")
        self._0 = corpus_0
        self._1 = corpus_1
        self._stats = Statistics(self)
        self._sim = Similarity(self)
        self._polarity = Polarity(self)

    @property
    def stats(self):
        return self._stats

    @property
    def sim(self):
        return self._sim

    @property
    def polarity(self):
        return self._polarity

    @property
    def num_corpus(self):
        return 2

    @property
    def corpus_0(self):
        return self._0

    @property
    def corpus_1(self):
        return self._1

    @property
    def corpora(self):
        return [self._0, self._1]

    def keywords(self, method: str):
        """ Extract and return the keywords of the two corpus ranked by frequency. """
        method_map = {
            'tf': TFKeywords,
            'tfidf': TFIDFKeywords
        }
        cls_kw = method_map.get(method, None)
        if cls_kw is None: raise ValueError(f"Only {method_map.keys()} methods are supported.")
        return cls_kw(self._0).extracted(), cls_kw(self._1).extracted()

    def lexical_diversity(self):
        """ Return the lexical diversity comparison.

        A smaller corpus will generally have higher lexical diversity.
        """
        ld = dict()
        for i, corpus in enumerate(self.corpora):
            ld[f"corpus_{i}"] = len(corpus.vocab) / np.log(corpus.num_terms)
        return ld
