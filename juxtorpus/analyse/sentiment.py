from typing import Optional
from juxtorpus.corpus import Corpus
from juxtorpus.corpus.meta import DocMeta, SeriesMeta
from juxtorpus.viz import Viz, Widget
import pandas as pd


def analyse_with_sentiment(corpus: Corpus, model: str, add_results: bool, results_prefix: str = 'sentiment'):
    sentiment = Sentiment(model=model)
    results = sentiment.run(corpus=corpus)
    if add_results: results.add_to_corpus(id_=results_prefix)
    return results


def _run_textblob(corpus: Corpus) -> pd.Series:
    from juxtorpus.corpus.processors import process
    import spacy
    from spacytextblob.spacytextblob import SpacyTextBlob
    nlp = spacy.load('en_core_web_sm')
    nlp.add_pipe('spacytextblob')
    scorpus = process(corpus, nlp=nlp)
    doc_meta: DocMeta = scorpus.meta.get('._.polarity')
    sentiments = scorpus.docs().apply(lambda doc: doc_meta._get_doc_attr(doc))
    return sentiments


_MODELS = {
    'textblob': _run_textblob
}


class SentimentResults(Viz):
    def __init__(self, corpus: Corpus, sentiments: pd.Series):
        self._corpus = corpus
        self._sentiments = sentiments

    @property
    def corpus(self) -> Corpus: return self._corpus

    @property
    def sentiments(self): return self._sentiments

    def add_to_corpus(self, id_: str):
        meta = SeriesMeta(id_=id_, series=self._sentiments)
        self._corpus.add_meta(meta)

    def render(self):
        print("Renders the sentiment analysis result.")
        # note: this requires the documents (corpus) and its sentiments.


class Sentiment(object):
    """ Sentiment Analysis
    Analyse the sentiment e.g. positive, neutral, negative on the documents in the corpus.
    """

    def __init__(self, model: str):
        self._model = model
        self._results: Optional[SentimentResults] = None

    @property
    def model(self) -> str:
        return self._model

    def run(self, corpus: Corpus) -> SentimentResults:
        func = _MODELS.get(self.model)
        sentiments = func(corpus=corpus)
        results = SentimentResults(corpus, sentiments)
        self._results = results
        return results

    def results(self) -> Optional[SentimentResults]:
        return self._results
