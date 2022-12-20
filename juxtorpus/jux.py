from juxtorpus.corpus import Corpus
from juxtorpus.stats import Statistics
from juxtorpus.features.similarity import Similarity
from juxtorpus.features.keywords import Keywords, RakeKeywords, TFKeywords, TFIDFKeywords
from juxtorpus.features.polarity import Polarity

import numpy as np
import pandas as pd
from typing import TypeVar

_CorpusT = TypeVar('_CorpusT', bound=Corpus)  # Corpus subclass


class Jux:
    """ Jux
    This is the main class for Juxtorpus. It takes in 2 corpus and exposes numerous functions
    to help contrast the two corpus.

    It is expected that the exposed functions are used as tools that serve as building blocks
    for your own further analysis.
    """

    def __init__(self, corpus_0: _CorpusT, corpus_1: _CorpusT):
        # NOTE: numeric variables are used to maintain consistency with column names in pandas
        self._0 = corpus_0
        self._1 = corpus_1
        self._stats = Statistics(self)
        self._sim = Similarity(self)

    @property
    def stats(self):
        return self._stats

    @property
    def sim(self):
        return self._sim

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

    def summary(self):
        return pd.concat([self._0.summary().rename('corpus_a'), self._1.summary().rename('corpus_b')], axis=1)

    @property
    def shares_parent(self) -> bool:
        return self._0.find_root() is self._1.find_root()

    def keywords(self, method: str):
        """ Extract and return the keywords of the two corpus ranked by frequency. """
        method_map = {
            'rake': RakeKeywords,
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
        ld_A = len(self._0.unique_terms) / np.log(self._0.num_terms)
        ld_B = len(self._1.unique_terms) / np.log(self._1.num_terms)
        return ld_A - ld_B, {'corpusA': ld_A, 'corpusB': ld_B}
