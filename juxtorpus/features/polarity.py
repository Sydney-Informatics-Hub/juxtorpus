""" Polarity

This module handles the calculations of polarity of terms based on a chosen metric.

Metrics:
1. term frequency (normalised on total terms)
2. tfidf
3. log likelihood

Output:
-> dataframe
"""

from typing import TYPE_CHECKING, Optional, Callable
import weakref as wr
import pandas as pd
from typing import Generator
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

    def tf(self, dtm_names: tuple[str, str]) -> pd.DataFrame:
        """ Uses the term frequency to produce the polarity score.

        Polarity = Corpus 0's tf - Corpus 1's tf.
        """
        jux = self._jux()
        corp_0, corp_1 = jux.corpus_0, jux.corpus_1
        dtm_0, dtm_1 = dtm_names
        dtm_0: DTM = corp_0.get_dtm(dtm_0)
        dtm_1: DTM = corp_1.get_dtm(dtm_1)

        df: pd.DataFrame = pd.concat([
            pd.Series(dtm_0.terms_vector, index=dtm_0.terms, name=f'{corp_0}_ftable'),
            pd.Series(dtm_1.terms_vector, index=dtm_1.terms, name=f'{corp_1}_ftable')
        ], axis=1).fillna(0)
        df['polarity'] = df[f"{corp_0}_ftable"] - df[f"{corp_0}_ftable"]
        return df

    def tfidf(self, tokeniser_func: Optional = None, lower=True):
        """ Uses the tfidf scores to produce the polarity score.

        Polarity = Corpus 0's tfidf - Corpus 1's tfidf.
        """
        dtms = self._selected_dtms(tokeniser_func, lower=lower)
        fts = (dtm.freq_table() for dtm in dtms)
        renamed_ft = [(f"{ft.name}_corpus_{i}", ft) for i, ft in enumerate(fts)]
        df = pd.concat([ft.series.rename(name) / ft.total for name, ft in renamed_ft], axis=1).fillna(0)
        df['polarity'] = df[renamed_ft[0][0]] - df[renamed_ft[1][0]]
        return df

    def log_likelihood(self, tokeniser_func: Optional = None):
        j = self._jux()
        llv = j.stats.log_likelihood_and_effect_size()
        tf_polarity = self.tf(tokeniser_func)['polarity']
        llv['polarity'] = (tf_polarity * llv['log_likelihood_llv']) / tf_polarity.abs()
        return llv

    def _selected_dtms(self, tokeniser_func: Optional, **kwargs) -> Generator[DTM, None, None]:
        """ Return a generator DTMs given a tokeniser function."""
        lower = kwargs.pop('lower', True)
        if tokeniser_func:
            return (corpus.create_custom_dtm(tokeniser_func, inplace=False, **kwargs) for corpus in self._jux().corpora)
        if not lower:  # note: corpus.dtm is lowered by default.
            return (corpus.create_custom_dtm(None, inplace=False, **kwargs) for corpus in self._jux().corpora)
        else:
            return (corpus.dtm for corpus in self._jux().corpora)

    def wordcloud(self, metric: str, top: int = 50, colours=('blue', 'red'), stopwords: list[str] = None,
                  tokeniser_func: Optional[Callable] = None, **kwargs):
        """ Generate a wordcloud using one of the 3 modes tf, tfidf, log_likelihood. """
        polarity_wordcloud_func = self.metrics.get(metric, None)
        if polarity_wordcloud_func is None:
            raise LookupError(f"Mode {metric} does not exist. Choose either {', '.join(self.metrics.keys())}")
        assert len(colours) == 2, "There can only be 2 colours. e.g. ('blue', 'red')."

        height, width = 24, 24
        pwc, add_legend = polarity_wordcloud_func(top, colours, tokeniser_func, stopwords, **kwargs)
        pwc._build(resolution_scale=int(height * width * 0.005))
        fig, ax = plt.subplots(figsize=(height / 2, width / 2))

        names = self._jux().corpus_0.name, self._jux().corpus_1.name
        legend_elements = [Patch(facecolor=colours[0], label=names[0]), Patch(facecolor=colours[1], label=names[1])]
        legend_elements.extend(add_legend)

        ax.imshow(pwc.wc, interpolation='bilinear')
        ax.legend(handles=legend_elements, prop={'size': 12}, loc='lower left', bbox_to_anchor=(1, 0.5))
        ax.axis('off')
        plt.tight_layout()  # Adjust the layout to prevent overlapping
        plt.show()

    def _wordcloud_tf(self, top: int, colours: tuple[str], tokeniser_func, stopwords: list[str] = None, **kwargs):
        assert len(colours) == 2, "Only supports 2 colours."
        if stopwords is None: stopwords = list()
        sw = stopwords
        sw.extend(ENGLISH_STOP_WORDS)

        df = self.tf(tokeniser_func, **kwargs)
        df = df[~df.index.isin(sw)]
        df['summed'] = df['freq_corpus_0'] + df['freq_corpus_1']
        df['polarity_div_summed'] = df['polarity'].abs() / df['summed']

        df_tmp = df.sort_values(by='summed', ascending=False).iloc[:top]

        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='polarity_div_summed')
        pwc.gradate(colours[0], colours[1])

        add_legend = [Patch(facecolor='None', label='Size: Polarised and Rare'),
                      Patch(facecolor='None', label='Solid: Higher frequency to one corpus'),
                      Patch(facecolor='None', label='Translucent: Similar frequency'), ]
        return pwc, add_legend

    def _wordcloud_tfidf(self, top: int, colours: tuple[str], tokeniser_func, stopwords: list[str] = None, **kwargs):
        assert len(colours) == 2, "Only supports 2 colours."
        if stopwords is None: stopwords = list()
        sw = stopwords
        sw.extend(ENGLISH_STOP_WORDS)

        df = self.tfidf(tokeniser_func, **kwargs)
        df['size'] = df.polarity.abs()
        df = df[~df.index.isin(sw)]
        df_tmp = df.sort_values(by='size', ascending=False).iloc[:top]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='size')
        pwc.gradate(colours[0], colours[1])

        add_legend = [Patch(facecolor='None', label='Size: Tfidf of both'),
                      Patch(facecolor='None', label='Solid: Higher Tfidf to one corpus'),
                      Patch(facecolor='None', label='Translucent: Similar tfidf')]
        return pwc, add_legend

    def _wordcloud_log_likelihood(self, top: int, colours: tuple[str], tokeniser_func,
                                  stopwords: list[str] = None,
                                  **kwargs):
        assert len(colours) == 2, "Only supports 2 colours."
        if stopwords is None: stopwords = list()
        sw = stopwords
        sw.extend(ENGLISH_STOP_WORDS)

        df = self.log_likelihood(tokeniser_func)
        tf_df = self.tf()
        df['summed'] = tf_df['freq_corpus_0'] + tf_df['freq_corpus_1']
        df['polarity_div_summed'] = df['polarity'].abs() / df['summed']
        df = df[~df.index.isin(sw)]
        df_tmp = df.sort_values(by='summed', ascending=False).iloc[:top]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='polarity_div_summed')
        pwc.gradate(colours[0], colours[1])

        add_legend = [Patch(facecolor='None', label='Size: Polarised and Rare'),
                      Patch(facecolor='None', label='Solid: Higher log likelihood to one corpus'),
                      Patch(facecolor='None', label='Translucent: Similar log likelihood')]
        return pwc, add_legend
