import re
import unittest

import pandas as pd
import numpy as np
from juxtorpus.corpus import Corpus
from juxtorpus.corpus.meta import SeriesMeta


def random_mask(corpus: Corpus):
    mask = pd.Series([np.random.choice([True, False]) for _ in range(len(corpus))], index=corpus._df.index)
    num_trues = mask.sum()
    return mask, num_trues


def is_equal(v1, v2):
    checks = v1 != v2  # comparing with == is less efficient than !=
    return not checks.data.any()


class TestCorpus(unittest.TestCase):
    def setUp(self) -> None:
        df = pd.read_csv('tests/assets/Geolocated_places_climate_with_LGA_and_remoteness_0.csv',
                         usecols=['processed_text', 'tweet_lga'])
        self.corpus = Corpus.from_dataframe(df, col_doc='processed_text')

    def test_Given_corpus_When_cloned_Then_cloned_parent_is_corpus(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned.parent == self.corpus, "Cloned parent is not properly inherited."

    def test_Given_corpus_When_cloned_twice_Then_final_clone_root_is_corpus(self):
        num_clones = 2
        for i in range(num_clones):
            mask, num_trues = random_mask(self.corpus)
            cloned = self.corpus.cloned(mask)
            assert cloned.find_root() == self.corpus, "Cloned is not the correct corpus."

    def test_Given_cloned_when_detach_Then_detached_is_root(self):
        # detached corpus should be root, DTM should be rebuilt.
        mask, _ = random_mask(self.corpus)
        clone = self.corpus.cloned(mask)

        orig_num_uniqs = self.corpus.dtm.shape[1]
        assert orig_num_uniqs == clone.dtm.shape[1], "Precondition not met for downstream assertion."

        detached = clone.detached()
        assert detached.is_root, "Clone is detached. It should be root."
        assert detached.dtm.shape[1] != orig_num_uniqs

    def test_Given_corpus_When_cloned_Then_subcorpus_size_equals_mask(self):
        """ Basic check of the cloned corpus. Texts, DTM and meta keys."""
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert len(cloned) == num_trues

    def test_Given_corpus_When_cloned_Then_cloned_meta_registry_exist_and_keys_inherited(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned.meta, "Meta registry is not cloned."
        assert set(cloned.meta.keys()) == set(self.corpus.meta.keys())

    def test_Given_scorpus_When_cloned_Then_cloned_dtm_registry_exist(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned._dtm_registry, "DTM Registry is not cloned."

    def test_Given_corpus_When_cloned_Then_subcorpus_dtm_size_equals_mask(self):
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        assert cloned.dtm.matrix.shape[0] == num_trues

    def test_Given_corpus_When_cloned_Then_normal_dtm_is_valid(self):
        """ Clone the corpus twice and ensure the root dtm index
        and adjusted subcorpus dtm index is equivalent each time. """
        mask, num_trues = random_mask(self.corpus)
        cloned = self.corpus.cloned(mask)
        # check if the cloned corpus dtm have the correct document term vectors
        # randomly chooses 5
        texts = cloned.docs()
        cloned_indices = np.random.randint(0, len(texts), size=5)
        for cloned_idx in cloned_indices:
            original_idx = texts.index[cloned_idx]
            assert is_equal(self.corpus.dtm.matrix[original_idx, :], cloned.dtm.matrix[cloned_idx, :])

        mask, num_trues = random_mask(cloned)
        cloned_again = cloned.cloned(mask)
        texts = cloned_again.docs()
        cloned_indices = np.random.randint(0, len(texts), size=5)
        for cloned_idx in cloned_indices:
            original_idx = texts.index[cloned_idx]
            assert is_equal(self.corpus.dtm.matrix[original_idx, :], cloned_again.dtm.matrix[cloned_idx, :])

    def test_Given_corpus_When_cloned_Then_cloned_custom_dtm_is_valid(self):
        mask, _ = random_mask(self.corpus)
        _ = self.corpus.create_custom_dtm(lambda text: re.findall(r'@\w+', text))  # function doesn't matter
        clone = self.corpus.cloned(mask)

        texts = clone.docs()
        clone_indices = np.random.randint(0, len(texts), size=5)
        for clone_idx in clone_indices:
            original_idx = texts.index[clone_idx]
            assert is_equal(self.corpus.custom_dtm.matrix[original_idx, :], clone.custom_dtm.matrix[clone_idx, :])

        mask, num_trues = random_mask(clone)
        clone_again = clone.cloned(mask)
        texts = clone_again.docs()
        clone_indices = np.random.randint(0, len(texts), size=5)
        for clone_idx in clone_indices:
            original_idx = texts.index[clone_idx]
            assert is_equal(self.corpus.custom_dtm.matrix[original_idx, :], clone_again.custom_dtm.matrix[clone_idx, :])

    def test_Given_corpus_When_to_dataframe_Then_size_are_matched(self):
        df = self.corpus.to_dataframe()
        assert len(df) == len(self.corpus), f"Mismatched df size: {len(self.corpus)=} {len(df)=}."

        mask, num_trues = random_mask(self.corpus)
        subcorpus = self.corpus.cloned(mask)
        df = subcorpus.to_dataframe()
        assert len(df) == len(subcorpus), f"Mismatched df size: {len(subcorpus)=} {len(df)=}."

    def test_Given_corpus_When_to_dataframe_Then_docs_are_added(self):
        df = self.corpus.to_dataframe()
        assert self.corpus.COL_DOC in df.columns, f"{self.corpus.COL_DOC} not found in exported dataframe."

        mask, num_trues = random_mask(self.corpus)
        subcorpus = self.corpus.cloned(mask)
        df = subcorpus.to_dataframe()
        assert subcorpus.COL_DOC in df.columns, f"{self.corpus.COL_DOC} not found in exported dataframe."

    def test_Given_corpus_When_to_dataframe_Then_series_metas_are_added(self):
        df = self.corpus.to_dataframe()
        cols = set(df.columns)
        doc_and_meta_ids = set([self.corpus.COL_DOC] +
                               [meta_id for meta_id, meta in self.corpus.meta.items() if isinstance(meta, SeriesMeta)])
        assert len(cols.intersection(doc_and_meta_ids)) == len(cols), \
            f"Invalid exported dataframe columns. Expecting {doc_and_meta_ids}. Got {cols}"

        mask, num_trues = random_mask(self.corpus)
        subcorpus = self.corpus.cloned(mask)
        df = subcorpus.to_dataframe()
        cols = set(df.columns)
        doc_and_meta_ids = set([subcorpus.COL_DOC] +
                               [meta_id for meta_id, meta in self.corpus.meta.items() if isinstance(meta, SeriesMeta)])
        assert len(cols.intersection(doc_and_meta_ids)) == len(cols), \
            f"Invalid exported dataframe columns. Expecting {doc_and_meta_ids}. Got {cols}"

    def test_Given_corpus_When_getitem_Then_correct_doc_is_returned(self):
        doc: str = self.corpus[0]
        assert doc == self.corpus.docs().iloc[0]

        start, stop = 4, 9
        docs: list[str] = self.corpus[start:stop]
        assert len(docs) == (9 - 4), \
            f"getitem returned the wrong number of documents. Expected: {9 - 4}. Got: {len(docs)}"

        mask, num_trues = random_mask(self.corpus)
        subcorpus = self.corpus.cloned(mask)
        doc: str = subcorpus[0]
        assert doc == subcorpus.docs().iloc[0]

        start, stop = 4, 9
        docs: list[str] = subcorpus[start:stop]
        assert len(docs) == (9 - 4), \
            f"getitem returned the wrong number of documents. Expected: {9 - 4}. Got: {len(docs)}"

    # Corpus name must be unique.
    # 1. create two corpus with the same name with __init__
    def test_Given_two_corpus_When_init_with_same_name_Then_one_is_amended(self):
        SHARED_NAME = 'shared_name'
        corpus = Corpus.from_dataframe(pd.DataFrame(['a'], columns=['t']), col_doc='t', name=SHARED_NAME)
        corpus2 = Corpus.from_dataframe(pd.DataFrame(['a'], columns=['t']), col_doc='t', name=SHARED_NAME)

        assert corpus.name != corpus2.name, "Corpus names must be unique."

    # 2. create a corpus and then clone it.
    def test_Given_corpus_When_cloned_Then_new_name_is_given(self):
        corpus = Corpus.from_dataframe(pd.DataFrame(['a'] * 100, columns=['t']), col_doc='t', name='root')

        existing_names = {corpus.name, }
        subcorpus = corpus
        for i in range(10):
            mask, _ = random_mask(subcorpus)
            subcorpus = subcorpus.cloned(mask)
            assert subcorpus.name not in existing_names, f"{subcorpus.name} already exist. Corpus names must be unique."
            existing_names.add(subcorpus.name)

    # 3. ensure number of names generated equals number of corpus.
    def test_Given_X_When_X_corpus_is_created_Then_X_names_exist(self):
        from juxtorpus.corpus.corpus import _ALL_CORPUS_NAMES
        SHARED_NAME = 'shared_name'
        corpus = Corpus.from_dataframe(pd.DataFrame(['a'] * 1000, columns=['t']), col_doc='t', name=SHARED_NAME)
        corpus2 = Corpus.from_dataframe(pd.DataFrame(['a'], columns=['t']), col_doc='t', name=SHARED_NAME)
        subcorpus = corpus
        for i in range(10):
            mask, _ = random_mask(subcorpus)
            subcorpus = subcorpus.cloned(mask)

        NUM_CORPUS_IN_SETUP_FN = 1
        NUM_CORPUS_IN_THIS_FN = 2 + 10
        assert len(_ALL_CORPUS_NAMES) == NUM_CORPUS_IN_SETUP_FN + NUM_CORPUS_IN_THIS_FN, \
            "Number of unique names created should equal the number of corpus created."
