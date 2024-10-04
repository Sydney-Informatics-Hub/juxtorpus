""" Polarity

This module handles the calculations of polarity of terms based on a chosen metric.

Metrics:
1. term frequency (normalised on total terms)
2. tfidf
3. log likelihood

Output:
-> dataframe
"""

from typing import TYPE_CHECKING
import weakref as wr

import numpy as np
import pandas as pd
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from atap_corpus.parts.dtm import DTM
from juxtorpus.viz.polarity_wordcloud import PolarityWordCloud

if TYPE_CHECKING:
    from juxtorpus import Jux


class Polarity(object):
    """ Polarity
    Gives a 'polarity' score of the two corpus based on token statistics.
    A polarity score uses 0 as the midline, a positive score means corpus 0 is dominant. And vice versa.
    e.g. tf would use the term frequencies to use as the polarity score.

    Each function will return a dataframe with a 'polarity' column and other columns with values that composes
    the 'polarity' score.
    """

    def __init__(self, jux: 'Jux'):
        self._jux: wr.ref['Jux'] = wr.ref(jux)
        self.metrics = {
            'tf': self._wordcloud_tf,
            'tfidf': self._wordcloud_tfidf,
            'log_likelihood': self._wordcloud_log_likelihood
        }

    def tf(self, dtm_names: tuple[str, str] | str) -> pd.DataFrame:
        """ Uses the term frequency to produce the polarity score.

        Polarity = Corpus 0's tf - Corpus 1's tf.
        """
        jux = self._jux()
        corp_0, corp_1 = jux.corpus_0, jux.corpus_1
        if isinstance(dtm_names, tuple):
            dtm_0, dtm_1 = dtm_names
        elif isinstance(dtm_names, str):
            dtm_0, dtm_1 = dtm_names, dtm_names
        else:
            raise TypeError("dtm_names must be either tuple[str, str] or str.")

        dtm_0: DTM = corp_0.get_dtm(dtm_0)
        dtm_1: DTM = corp_1.get_dtm(dtm_1)

        df: pd.DataFrame = pd.concat([
            pd.Series(dtm_0.terms_vector/dtm_0.total, index=dtm_0.terms, name=f'{corp_0.name}_tf'),
            pd.Series(dtm_1.terms_vector/dtm_1.total, index=dtm_1.terms, name=f'{corp_1.name}_tf')
        ], axis=1).fillna(0)
        df['polarity'] = df[f"{corp_0.name}_tf"] - df[f"{corp_1.name}_tf"]
        return df

    def tfidf(self, dtm_names: tuple[str, str] | str):
        """ Uses the tfidf scores to produce the polarity score.

        Polarity = Corpus 0's tfidf - Corpus 1's tfidf.
        """
        jux = self._jux()
        corp_0, corp_1 = jux.corpus_0, jux.corpus_1
        if isinstance(dtm_names, tuple):
            dtm_0, dtm_1 = dtm_names
        elif isinstance(dtm_names, str):
            dtm_0, dtm_1 = dtm_names, dtm_names
        else:
            raise TypeError("dtm_names must be either tuple[str, str] or str.")

        dtm_0: DTM = corp_0.get_dtm(dtm_0)
        dtm_1: DTM = corp_1.get_dtm(dtm_1)

        df: pd.DataFrame = pd.concat([
            pd.Series(dtm_0.terms_vector/dtm_0.total, index=dtm_0.terms, name=f'{corp_0.name}_tf'),
            pd.Series(dtm_1.terms_vector/dtm_1.total, index=dtm_1.terms, name=f'{corp_1.name}_tf'),
            pd.Series(np.log10( dtm_0.shape[0]/np.minimum(dtm_0.matrix.toarray(), 1).sum(axis=0) ), index=dtm_0.terms, name=f'{corp_0.name}_df'),
            pd.Series(np.log10( dtm_1.shape[0]/np.minimum(dtm_1.matrix.toarray(), 1).sum(axis=0) ), index=dtm_1.terms, name=f'{corp_1.name}_df'),
        ], axis=1).fillna(0)
        df[f'{corp_0.name}_tfidf'] = df[f'{corp_0.name}_tf'] * df[f'{corp_0.name}_df']
        df[f'{corp_1.name}_tfidf'] = df[f'{corp_1.name}_tf'] * df[f'{corp_1.name}_df']
        df['polarity'] = df[f'{corp_0.name}_tfidf'] - df[f'{corp_1.name}_tfidf']
        return df

    def log_likelihood(self, dtm_names: tuple[str, str] | str):
        j = self._jux()
        llv = j.stats.log_likelihood_and_effect_size(dtm_names)
        tf_polarity = self.tf(dtm_names)['polarity']
        llv['polarity'] = (tf_polarity * llv['log_likelihood_llv']) / tf_polarity.abs()
        return llv

    def wordcloud(self, dtm_names: tuple[str, str] | str, metric: str, top: int = 50, colours=('blue', 'red'),
                  stopwords: list[str] = None, return_wc: bool = False, **kwargs):
        """ Generate a wordcloud using one of the 3 modes tf, tfidf, log_likelihood. """
        polarity_wordcloud_func = self.metrics.get(metric, None)
        if polarity_wordcloud_func is None:
            raise LookupError(f"Mode {metric} does not exist. Choose either {', '.join(self.metrics.keys())}")
        assert len(colours) == 2, "There can only be 2 colours. e.g. ('blue', 'red')."

        height, width = 24, 24
        pwc, add_legend = polarity_wordcloud_func(dtm_names, top, colours, stopwords, **kwargs)
        pwc._build(resolution_scale=int(height * width * 0.005))

        if return_wc:
            return pwc
        fig, ax = plt.subplots(figsize=(height / 2, width / 2))
        names = self._jux().corpus_0.name, self._jux().corpus_1.name
        legend_elements = [Patch(facecolor=colours[0], label=names[0]), Patch(facecolor=colours[1], label=names[1])]
        legend_elements.extend(add_legend)

        ax.imshow(pwc.wc, interpolation='bilinear')
        ax.legend(handles=legend_elements, prop={'size': 12}, loc='lower left', bbox_to_anchor=(1, 0.5))
        ax.axis('off')
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.show()

    def _wordcloud_tf(self, dtm_names: tuple[str, str] | str, top: int, colours: tuple[str],
                      stopwords: list[str] = None, **kwargs):
        assert len(colours) == 2, "Only supports 2 colours."
        if stopwords is None: stopwords = list()
        sw = stopwords
        sw.extend(ENGLISH_STOP_WORDS)

        corp_0, corp_1 = self._jux().corpus_0, self._jux().corpus_1

        df = self.tf(dtm_names, **kwargs)
        df = df[~df.index.isin(sw)]
        df['summed'] = df[f'{corp_0.name}_tf'] + df[f'{corp_1.name}_tf']
        df['polarity_div_summed'] = df['polarity'].abs() / df['summed']

        df_tmp = df.sort_values(by='summed', ascending=False).iloc[:top]

        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='polarity_div_summed')
        pwc.gradate(colours[0], colours[1])

        add_legend = [Patch(facecolor='None', label='Size: Polarised and Rare'),
                      Patch(facecolor='None', label='Solid: Higher frequency to one corpus'),
                      Patch(facecolor='None', label='Translucent: Similar frequency'), ]
        return pwc, add_legend

    def _wordcloud_tfidf(self, dtm_names: tuple[str, str] | str, top: int, colours: tuple[str],
                         stopwords: list[str] = None, **kwargs):
        assert len(colours) == 2, "Only supports 2 colours."
        if stopwords is None:
            sw = list(ENGLISH_STOP_WORDS)
        else:
            sw = stopwords

        df = self.tfidf(dtm_names, **kwargs)
        df['size'] = df['polarity'].abs()
        df = df[~df.index.isin(sw)]
        df = df[df['polarity'].notna()]
        df_tmp = df.sort_values(by='size', ascending=False).iloc[:top]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='size')
        pwc.gradate(colours[0], colours[1])

        add_legend = [Patch(facecolor='None', label='Size: Tfidf of both'),
                      Patch(facecolor='None', label='Solid: Higher Tfidf to one corpus'),
                      Patch(facecolor='None', label='Translucent: Similar tfidf')]
        return pwc, add_legend

    def _wordcloud_log_likelihood(self, dtm_names: tuple[str, str] | str, top: int, colours: tuple[str],
                                  stopwords: list[str] = None):
        assert len(colours) == 2, "Only supports 2 colours."
        if stopwords is None: stopwords = list()
        sw = stopwords
        sw.extend(ENGLISH_STOP_WORDS)

        jux = self._jux()
        corp_0, corp_1 = jux.corpus_0, jux.corpus_1

        df = self.log_likelihood(dtm_names)
        tf_df = self.tf(dtm_names)
        df['summed'] = tf_df[f"{corp_0.name}_tf"] + tf_df[f"{corp_1.name}_tf"]
        df['polarity_div_summed'] = df['polarity'].abs() / df['summed']
        df = df[~df.index.isin(sw)]
        df_tmp = df.sort_values(by='summed', ascending=False).iloc[:top]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='polarity_div_summed')
        pwc.gradate(colours[0], colours[1])

        add_legend = [Patch(facecolor='None', label='Size: Polarised and Rare'),
                      Patch(facecolor='None', label='Solid: Higher log likelihood to one corpus'),
                      Patch(facecolor='None', label='Translucent: Similar log likelihood')]
        return pwc, add_legend
