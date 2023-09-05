""" test.tm.topsbm.py
"""

from unittest import TestCase
from pathlib import Path

import pandas as pd
from juxtorpus.corpus import Corpus, Corpora
from juxtorpus.corpus.meta import SeriesMeta
from juxtorpus.tm import TopSBM

ASSET_DIR = Path("./tests/assets/topsbm")


class TestTopSBM(TestCase):
    def setUp(self) -> None:
        assert ASSET_DIR.is_dir(), "Missing test assets path."
        files = set([p.name for p in ASSET_DIR.glob("*")])
        assert "corpus.txt" in files, "Missing corpus.txt."
        assert "titles.txt" in files, "Missing titles.txt."
        titles = pd.read_table("./topsbm/titles.txt", header=None)
        corpus = Corpus.from_dataframe(pd.read_table("./topsbm/corpus.txt", header=None), col_doc=0)
        self.corpus = corpus

        corpus.add_meta(SeriesMeta(id_='title', series=titles.loc[:, 0]))
        self.meta_title_id = 'title'

    def test_added_meta_is_correct(self):
        """ Ensures the added topic meta is correct.
        1. correct number topics are added onto the corpus.
        2. each topic's distribution is correct.
        """
        num_metas = len(self.corpus.meta.keys())
        topsbm = TopSBM(self.corpus, meta_title=self.meta_title_id)
        topsbm.build()
        added = topsbm.add_results_to_corpus()
        diff_metas = len(self.corpus.meta.keys()) - num_metas
        assert diff_metas == len(topsbm.topics), "Added more topics than inferred by TopSBM."
        for i, meta_id in enumerate(added.get('meta_ids')):
            topsbm_dist = topsbm.doc_topics.iloc[:, i]
            corpus_dist = self.corpus.meta.get(meta_id).series
            assert (topsbm_dist == corpus_dist).all(), f"Mismatched TopSBM docs-topic dist and {meta_id} dist."

    def test_added_dtm_is_correct(self):
        """ Ensures the added doc-topic DTM is correct. """
        num_dtms = len(self.corpus._dtm_registry.keys())
        topsbm = TopSBM(self.corpus, meta_title=self.meta_title_id)
        topsbm.build()
        added = topsbm.add_results_to_corpus()
        diff_dtms = len(self.corpus._dtm_registry.keys()) - num_dtms
        assert diff_dtms == 1, "Added more than 1 custom DTM to corpus."
        custom_dtm_id = added.get('custom_dtm_id')
        cdtm = self.corpus._dtm_registry.get_or_raise_err(custom_dtm_id)

        fp_imprecision = 0.01
        topsbm_sum = topsbm.doc_topics.sum().sum()
        assert topsbm_sum * (1 - fp_imprecision) < cdtm.matrix.sum() < topsbm_sum * (1 + fp_imprecision), \
            f"Mismatched TopSBM docs-topic values and {custom_dtm_id} values."

    def test_added_dtm_clones(self):
        """ Ensures added results back to corpus clones properly.
        In retrospect, this should really be a test for Corpus but doesn't hurt to have redundancy here.
        """
        topsbm = TopSBM(self.corpus, meta_title=self.meta_title_id)
        topsbm.build()
        added = topsbm.add_results_to_corpus()
        dtm = self.corpus._dtm_registry.get_or_raise_err(added.get('custom_dtm_id'))
        topic_0 = self.corpus.slicer.filter_by_range('TopSBM_topic-0', min_=0.1)
        cloned_dtm = topic_0._dtm_registry.get_or_raise_err(added.get('custom_dtm_id'))
        assert len(topic_0) == cloned_dtm.shape[0], "Mismatched number of documents after cloning TopSBM custom dtm."
        assert dtm.shape[1] == cloned_dtm.shape[1], "Mismatched number of terms after cloning TopSBM custom dtm."
