"""
Similarity between 2 Corpus.

1. jaccard similarity
2. pca similarity
"""

from typing import TYPE_CHECKING, Union
import weakref as wr

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from juxtorpus.constants import CORPUS_ID_COL_NAME_FORMAT

if TYPE_CHECKING:
    from juxtorpus import Jux


def _cos_sim(v0: Union[np.ndarray, pd.Series], v1: Union[np.ndarray, pd.Series]):
    if isinstance(v0, np.ndarray) and isinstance(v1, np.ndarray):
        assert v0.ndim == 1 and v1.ndim == 1, "Must be 1d array."
        assert v0.shape[0] == v1.shape[0], f"Mismatched shape {v0.shape=} {v1.shape=}"
        if v0.shape[0] == 0: return 0
    elif isinstance(v0, pd.Series) and isinstance(v1, pd.Series):
        assert len(v0) == len(v1), f"Mismatched shape {len(v0)=} {len(v1)=}"
        if len(v0) == 0: return 0
    else:
        raise ValueError(f"They must both be either "
                         f"{np.ndarray.__class__.__name__} or "
                         f"{pd.Series.__class__.__name__}.")
    return 1 - distance.cosine(v0, v1)


class Similarity(object):
    def __init__(self, jux: 'Jux'):
        self._jux = wr.ref(jux)

    def jaccard(self, dtm_names: tuple[str] | str):
        """ Return a similarity score between the 2 corpus."""
        if isinstance(dtm_names, tuple):
            d1, d2 = dtm_names
        elif isinstance(dtm_names, str):
            d1 = d2 = dtm_names
        else:
            raise NotImplementedError(f"dtm_names must be either a tuple of str or str.")

        jux = self._jux()
        u0: set[str] = set(jux.corpus_0.dtms[d1].vocab(nonzero=True))
        u1: set[str] = set(jux.corpus_1.dtms[d2].vocab(nonzero=True))
        inter, union = len(u0.intersection(u1)), len(u0.union(u1))
        if union == 0: return 0
        if union < 0: raise RuntimeError("Union is < 0, this should not happen.")
        if inter < 0: raise RuntimeError("Intersection is < 0, this should not happen.")
        return len(u0.intersection(u1)) / len(u0.union(u1))

    def lsa_pairwise_cosine(self, dtm_names: tuple[str] | str, n_components: int = 100, verbose=False):
        """ Decompose DTM to SVD and return the pairwise cosine similarity of the right singular matrix.

        Note: this may be different to the typical configuration using a TDM instead of DTM.
        However, sklearn only exposes the right singular matrix.
        tdm.T = (U Sigma V.T).T = V.T.T Sigma.T U.T = V Sigma U.T
        the term-topic matrix of U is now the right singular matrix if we use DTM instead of TDM.
        """
        if isinstance(dtm_names, tuple):
            d1, d2 = dtm_names
        elif isinstance(dtm_names, str):
            d1 = d2 = dtm_names
        else:
            raise NotImplementedError(f"dtm_names must be either a tuple of str or str.")
        A, B = self._jux().corpus_0, self._jux().corpus_1
        svd_A = TruncatedSVD(n_components=n_components).fit(A.dtms[d1].matrix)
        svd_B = TruncatedSVD(n_components=n_components).fit(B.dtms[d2].matrix)
        top_topics = 5
        if verbose:
            top_terms = 5
            for corpus, svd in [(A, svd_A), (B, svd_B)]:
                feature_indices = svd.components_.argsort()[::-1][
                                  :top_topics]  # highest value term in term-topic matrix
                terms = corpus.dtm.term_names[feature_indices]
                for i in range(feature_indices.shape[0]):
                    print(f"Corpus {str(corpus)}: Singular columns [{i}] {terms[i][:top_terms]}")

        # pairwise cosine
        return cosine_similarity(svd_A.components_[:top_topics], svd_B.components_[:top_topics])

    # def cosine_similarity(self, metric: str, *args, **kwargs):
    #     metric_map = {
    #         'tf': self._cos_sim_tf,
    #         'tfidf': self._cos_sim_tfidf,
    #         'log_likelihood': self._cos_sim_llv
    #     }
    #     sim_fn = metric_map.get(metric, None)
    #     if sim_fn is None: raise ValueError(f"Only metrics {metric_map.keys()} are supported.")
    #     return sim_fn(*args, **kwargs)
    #
    # def _cos_sim_llv(self, baseline: pd.DataFrame = None):
    #     res = self._jux().stats.log_likelihood_and_effect_size(baseline=baseline).fillna(0)
    #     return _cos_sim(res[CORPUS_ID_COL_NAME_FORMAT.format('log_likelihood_llv', 0)],
    #                     res[CORPUS_ID_COL_NAME_FORMAT.format('log_likelihood_llv', 1)])
    #
    # def _cos_sim_tf(self, without: list[str] = None) -> float:
    #     seriess = list()
    #     for i, corpus in enumerate(self._jux().corpora):
    #         if len(corpus) <= 0: return 0.0
    #         ft = corpus.dtm.freq_table(nonzero=True)
    #         if without: ft.remove(without)
    #         seriess.append(ft.series.rename(CORPUS_ID_COL_NAME_FORMAT.format(ft.name, i)))
    #
    #     res = pd.concat(seriess, axis=1).fillna(0)
    #     return _cos_sim(res.iloc[:, 0], res.iloc[:, 1])
    #
    # def _cos_sim_tfidf(self, **kwargs):
    #     seriess = list()
    #     for i, corpus in enumerate(self._jux().corpora):
    #         if len(corpus) <= 0: return 0.0
    #         ft = corpus.dtm.tfidf(**kwargs).freq_table(nonzero=True)
    #         seriess.append(ft.series.rename(CORPUS_ID_COL_NAME_FORMAT.format(ft.name, i)))
    #     res = pd.concat(seriess, axis=1).fillna(0)
    #     return _cos_sim(res.iloc[:, 0], res.iloc[:, 1])
