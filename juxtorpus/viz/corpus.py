""" Collection of Visualisation functions for Corpus

"""
from wordcloud import WordCloud
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import math
from typing import Callable


def wordclouds(corpora, names: list[str],
               max_words: int = 50,
               metric: str = 'tf',
               word_type: str = 'word',
               stopwords: list[str] = None,
               lower: bool = True):
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
                        word_type=word_type,
                        stopwords=stopwords,
                        lower=lower)
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


def wordcloud(corpus, metric: str = 'tf', max_words: int = 50, word_type: str = 'word',
              stopwords: list[str] = None, lower: bool = True):
    wc = _wordcloud(corpus, max_words, metric, word_type, stopwords, lower)
    # h, w = 12, 12 * 1.5
    h, w = 6, 10
    plt.figure(figsize=(h, w))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def _wordcloud(corpus, max_words: int, metric: str, word_type: str, stopwords: list[str] = None, lower: bool = True):
    if stopwords is None: stopwords = list()
    stopwords.extend(ENGLISH_STOP_WORDS)
    word_types = {'word', 'hashtag', 'mention'}
    metrics = {'tf', 'tfidf'}
    assert word_type in word_types, f"{word_type} not in {', '.join(word_types)}"
    assert metric in metrics, f"{metric} not in {', '.join(metrics)}"
    wc = WordCloud(background_color='white', max_words=max_words, height=600, width=1200, stopwords=stopwords)

    def lower_wrapper(gen) -> Callable:
        def generate_lowered(doc):
            return (str(x).lower() for x in gen(doc))

        return generate_lowered

    if word_type == 'word':
        dtm = corpus.dtm  # corpus dtm is always lower cased.
    elif word_type == 'hashtag':
        gen = corpus._gen_hashtags_from
        if lower: gen = lower_wrapper(gen)
        dtm = corpus.create_custom_dtm(tokeniser_func=gen, inplace=False)
    elif word_type == 'mention':
        gen = corpus._gen_mentions_from
        if lower: gen = lower_wrapper(gen)
        dtm = corpus.create_custom_dtm(tokeniser_func=gen, inplace=False)
    else:
        raise ValueError(f"Word type {word_type} is not supported. Must be one of {', '.join(word_types)}")

    if metric == 'tf':
        with dtm.without_terms(stopwords) as dtm:
            counter = dtm.freq_table().series.to_dict()
            wc.generate_from_frequencies(counter)
            return wc
    elif metric == 'tfidf':
        with dtm.tfidf().without_terms(stopwords) as dtm:
            counter = dtm.tfidf().freq_table().series.to_dict()
            wc.generate_from_frequencies(counter)
            return wc
    else:
        raise ValueError(f"Metric {metric} is not supported. Must be one of {', '.join(metrics)}")


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
