""" Collection of Visualisation functions for Corpus

"""
from wordcloud import WordCloud
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import math
from typing import Callable
from tmtoolkit.bow.bow_stats import tfidf


def wordclouds(corpora, names: list[str],
               max_words: int = 50,
               metric: str = 'tf',
               dtm_name: str = 'tokens',
               stopwords: list[str] = None):
    MAX_COLS = 2
    nrows = math.ceil(len(names) / 2)
    fig, axes = plt.subplots(nrows=nrows, ncols=MAX_COLS, figsize=(16, 16 * 1.5))
    r, c = 0, 0
    for name in names:
        assert corpora[name], f"{name} does not exist in Corpora."
        corpus = corpora[name]
        wc = _wordcloud(corpus,
                        max_words=max_words,
                        metric=metric,
                        dtm_name=dtm_name,
                        stopwords=stopwords,
                        return_wc=False)
        if nrows == 1:
            ax = axes[c]
        else:
            ax = axes[r][c]
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        if c == MAX_COLS - 1: r += 1
        c = (c + 1) % MAX_COLS

    plt.tight_layout(pad=0)
    plt.show()


def wordcloud(corpus, metric: str = 'tf', max_words: int = 50, dtm_name: str = 'tokens',
              stopwords: list[str] = None, return_wc: bool = False):
    wc = _wordcloud(corpus, max_words, metric, dtm_name, stopwords)
    if return_wc:
        return wc
    # h, w = 12, 12 * 1.5
    h, w = 6, 10
    plt.figure(figsize=(h, w))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()

def dtm2tfidf(dtm):
    tfidf_mat = tfidf(dtm.matrix)
    freq = dict(zip(dtm.vocab(), sum(tfidf_mat.toarray())))
    return freq

def _wordcloud(corpus, max_words: int, metric: str, dtm_name: str, stopwords: list[str] = None):
    if stopwords is None: stopwords = list()
    # stopwords.extend(ENGLISH_STOP_WORDS)
    metrics = {'tf', 'tfidf'}
    assert dtm_name in corpus.dtms.keys(), f"{dtm_name} not in {', '.join(corpus.dtms.keys())}"
    assert metric in metrics, f"{metric} not in {', '.join(metrics)}"
    wc = WordCloud(background_color='white', max_words=max_words, height=600, width=1200, stopwords=stopwords)

    dtm = corpus.dtms[dtm_name]
    with dtm.without_terms(stopwords) as dtm:
        if metric == 'tf':
            counter = dtm.freq_table().to_dict()
        elif metric == 'tfidf':
            counter = dtm2tfidf(dtm)
        else:
            raise ValueError(f"Metric {metric} is not supported. Must be one of {', '.join(metrics)}")
    wc.generate_from_frequencies(counter)
    return wc


def timeline(corpus, datetime_meta: str, freq: str, meta_name: list[str] = None):
    time_meta = corpus.meta.get_or_raise_err(datetime_meta)
    if meta_name == None:
        meta_name = ['']
    if isinstance(meta_name, str):
        meta_name = [meta_name]
    assert pd.api.types.is_datetime64_any_dtype(time_meta.series), f"{time_meta.id} is not a datetime meta."
    fig = go.Figure()
    for name in meta_name:
        if name:
            s = pd.Series(corpus.meta.get_or_raise_err(name).series.tolist(), index=time_meta.series)
        else:
            s = pd.Series(time_meta.series.index, index=time_meta.series)
        s = s.groupby(pd.Grouper(level=0, freq=freq)).nunique(dropna=True)

        fig.add_trace(
            go.Scatter(x=s.index.tolist(), y=s.tolist(), name=name, showlegend=True)
        )
    freq_to_label = {'w': 'Week', 'm': 'Month', 'y': 'Year', 'd': 'Day'}
    key = freq.strip()[-1].lower()

    title = f"Count by {freq_to_label.get(key, key)}"
    xaxis_title, yaxis_title = f"{freq_to_label.get(key, key)}", "Count"
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig


def timelines(corpora, names: list[str], datetime_meta: str, freq: str, meta_name: str = ''):
    # datetime_series = None
    for name in names:
        corpus = corpora[name]
        assert corpus, f"{name} does not exist in corpora."
        # meta = corpus.meta.get_or_raise_err(datetime_meta)
        # if not datetime_series: datetime_series = meta.series
    fig = go.Figure()
    for name in names:
        time_meta = corpora[name].meta.get_or_raise_err(datetime_meta)
        if meta_name:
            s = pd.Series(corpora[name].meta.get_or_raise_err(meta_name).series.tolist(), index=time_meta.series)
        else:
            s = pd.Series(time_meta.series.index, index=time_meta.series)
        s = s.groupby(pd.Grouper(level=0, freq=freq)).nunique(dropna=True)
        fig.add_trace(
            go.Scatter(x=s.index.tolist(), y=s.tolist(), name=name, showlegend=True)
        )

    freq_to_label = {'w': 'Week', 'm': 'Month', 'y': 'Year', 'd': 'Day'}
    key = freq.strip()[-1].lower()
    f = freq.strip()[0]

    title = f"Count by {f} {freq_to_label.get(key, key)}"
    xaxis_title, yaxis_title = f"{freq_to_label.get(key, key)}", "Count"
    fig.update_layout(title=title, xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    return fig
