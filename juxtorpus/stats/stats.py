import weakref
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from .loglikelihood_effectsize import log_likelihood_and_effect_size
from juxtorpus.constants import CORPUS_ID_COL_NAME_FORMAT

if TYPE_CHECKING:
    from juxtorpus import Jux


class Statistics(object):
    def __init__(self, jux: 'Jux'):
        self._jux = weakref.ref(jux)

    def log_likelihood_and_effect_size(self, dtm_names: tuple[str] | str,
                                       baseline: pd.DataFrame | None = None) -> pd.DataFrame:
        if isinstance(dtm_names, tuple):
            d1, d2 = dtm_names
        elif isinstance(dtm_names, str):
            d1 = d2 = dtm_names
        else:
            raise NotImplementedError(f"dtm_names must be either a tuple of str or str.")

        c0, c1 = self._jux().corpus_0, self._jux().corpus_1
        dtm0, dtm1 = c0.dtms[d1], c1.dtms[d2]
        indices: list[np.ndarray] = [np.array(dtm0.terms)[dtm0.terms_vector.nonzero()[0]],
                                     np.array(dtm1.terms)[dtm1.terms_vector.nonzero()[0]]]
        values: list[np.ndarray] = [dtm0.terms_vector[dtm0.terms_vector.nonzero()[0]],
                                    dtm1.terms_vector[dtm1.terms_vector.nonzero()[0]]]
        ftables = [pd.Series(v, index=index) for v, index in zip(values, indices)]

        if baseline is None:
            res = log_likelihood_and_effect_size(ftables)
            res = res.filter(regex=r'(log_likelihood_llv|bayes_factor_bic|effect_size_ell)')
        else:
            res_list = list()
            for i, ft in enumerate(ftables):
                res = log_likelihood_and_effect_size([ft, baseline])
                res = res.filter(regex=r'(log_likelihood_llv|bayes_factor_bic|effect_size_ell)')
                mapper = dict()
                for col in res.columns:
                    mapper[col] = CORPUS_ID_COL_NAME_FORMAT.format(col, i)
                res.rename(mapper=mapper, axis=1, inplace=True)
                res_list.append(res)
            res = pd.concat(res_list, axis=1)
        return res
