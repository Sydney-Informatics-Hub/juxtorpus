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

from juxtorpus.corpus.dtm import DTM
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
        self.modes = {
            'tf': self._wordcloud_tf,
            'tfidf': self._wordcloud_tfidf,
            'log_likelihood': self._wordcloud_log_likelihood
        }

    def tf(self, tokeniser_func: Optional = None):
        """ Uses the term frequency to produce the polarity score.

        Polarity = Corpus 0's tf - Corpus 1's tf.
        """
        dtms = self._selected_dtms(tokeniser_func)
        fts = (dtm.freq_table() for dtm in dtms)

        renamed_ft = [(f"{ft.name}_corpus_{i}", ft) for i, ft in enumerate(fts)]
        df = pd.concat([ft.series.rename(name) / ft.total for name, ft in renamed_ft], axis=1).fillna(0)
        df['polarity'] = df[renamed_ft[0][0]] - df[renamed_ft[1][0]]
        return df

    def tfidf(self, tokeniser_func: Optional = None):
        """ Uses the tfidf scores to produce the polarity score.

        Polarity = Corpus 0's tfidf - Corpus 1's tfidf.
        """
        dtms = self._selected_dtms(tokeniser_func)
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

    def _selected_dtms(self, tokeniser_func: Optional) -> Generator[DTM, None, None]:
        """ Return a generator DTMs given a tokeniser function."""
        if tokeniser_func:
            return (corpus.create_custom_dtm(tokeniser_func) for corpus in self._jux().corpora)
        else:
            return (corpus.dtm for corpus in self._jux().corpora)

    def wordcloud(self, mode: str, tokeniser_func: Optional[Callable] = None, colours=('blue', 'red')):
        polarity_wordcloud_func = self.modes.get(mode, None)
        if polarity_wordcloud_func is None:
            raise LookupError(f"Mode {mode} does not exist. Choose either {', '.join(self.modes.keys())}")
        assert len(colours) == 2, "You may only "
        polarity_wordcloud_func(tokeniser_func).render(16, 16)

    def _wordcloud_tf(self, tokeniser_func):
        from nltk.corpus import stopwords
        sw = stopwords.words('english')

        df = self.tf(tokeniser_func)
        df = df[~df.index.isin(sw)]
        df['summed'] = df['freq_corpus_0'] + df['freq_corpus_1']
        df['polarity_div_summed'] = df['polarity'].abs() / df['summed']

        top = 50
        df_tmp = df.sort_values(by='summed', ascending=False).iloc[:top]

        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='polarity_div_summed')
        pwc.gradate('blue', 'red')
        return pwc

    def _wordcloud_tfidf(self, tokeniser_func):
        df = self.tfidf(tokeniser_func)
        df['size'] = df.polarity.abs()
        df_tmp = df.sort_values(by='size', ascending=False).iloc[:30]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='size')
        pwc.gradate('blue', 'red')
        return pwc

    def _wordcloud_log_likelihood(self, tokeniser_func):
        from nltk.corpus import stopwords
        sw = stopwords.words('english')

        df = self.log_likelihood(tokeniser_func)
        tf_df = self.tf()
        df['size'] = tf_df['freq_corpus_0'] + tf_df['freq_corpus_1']
        df = df[~df.index.isin(sw)]
        df_tmp = df.sort_values(by='size', ascending=False).iloc[:30]
        pwc = PolarityWordCloud(df_tmp, col_polarity='polarity', col_size='size')
        pwc.gradate('blue', 'red')
        return pwc
