import math

import nltk

from rake_nltk import Rake
from abc import ABCMeta, abstractmethod
from typing import Tuple, List, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import binarize
from scipy.sparse import csr_matrix
from collections import Counter
from spacy.matcher import Matcher
import numpy as np

from juxtorpus.corpus import Corpus, SpacyCorpus
from juxtorpus.matchers import no_stopwords, is_word, no_puncs_no_stopwords


class Keywords(metaclass=ABCMeta):
    def __init__(self, corpus: Corpus):
        self.corpus = corpus

    @abstractmethod
    def extracted(self) -> List[str]:
        raise NotImplemented("You are calling from the base class. Use one of the concrete ones.")


class RakeKeywords(Keywords):
    """ Implementation of Keywords extraction using Rake.
    package: https://pypi.org/project/rake-nltk/
    paper: https://www.researchgate.net/profile/Stuart_Rose/publication/227988510_Automatic_Keyword_Extraction_from_Individual_Documents/links/55071c570cf27e990e04c8bb.pdf

    RAKE begins keyword extraction on a document by parsing its text into a set of candidate keywords.
    First, the document text is split into an array of words by the specified word delimiters.
    This array is then split into sequences of contiguous words at phrase delimiters and stop word positions.
    Words within a sequence are assigned the same position in the text and together are considered a candidate keyword.
    """

    def extracted(self):
        _kw_A = Counter(RakeKeywords._rake(sentences=self.corpus.texts().tolist()))
        return _kw_A.most_common(20)

    @staticmethod
    def _rake(sentences: List[str]):
        import nltk
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')

        r = Rake()
        r.extract_keywords_from_sentences(sentences)
        return r.get_ranked_phrases()


class TFIDFKeywords(Keywords):
    def __init__(self, corpus: Corpus):
        super().__init__(corpus)
        self.count_vec = CountVectorizer(
            tokenizer=TFIDFKeywords._do_nothing,
            preprocessor=TFIDFKeywords._preprocess,
            ngram_range=(1, 1)  # default = (1,1)
        )

    def extracted(self):
        corpus_tfidf = self._corpus_tf_idf(smooth=False)
        keywords = [(word, corpus_tfidf[0][i]) for i, word in enumerate(self.count_vec.get_feature_names_out())]
        keywords.sort(key=lambda w_tfidf: w_tfidf[1], reverse=True)
        return keywords
        # return TFIDFKeywords._max_tfidfs(self.corpus)

    def _corpus_tf_idf(self, smooth: bool = False):
        """ Term frequency is of the entire corpus. Idfs calculated as per normal. """
        tfs = self.count_vec.fit_transform(self.corpus.texts())
        idfs = binarize(tfs, threshold=0.99)
        if smooth:
            pass  # TODO: smoothing of idfs using log perhaps.
        return np.array(csr_matrix.sum(tfs, axis=0) / csr_matrix.sum(idfs, axis=0))

    @staticmethod
    def _max_tfidfs(corpus: Corpus):
        # get the tfidf score of the docs.
        # get the tfidf score of each word and rank them that way.
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus.texts())
        col_words = vectorizer.get_feature_names_out()
        max_tfidf_cols = [(col_words[i], X[:, i].max()) for i in range(X.shape[1])]
        max_tfidf_cols.sort(key=lambda t: t[1], reverse=True)
        return max_tfidf_cols

    @staticmethod
    def _do_nothing(doc):
        """ Used to override default tokenizer and preprocessors in sklearn transformers."""
        return doc

    @staticmethod
    def _preprocess(doc):
        """ Filters punctuations and normalise case."""
        return [doc[start:end].text.lower() for _, start, end in no_puncs_no_stopwords(nlp.vocab)(doc)]


class TFKeywords(Keywords):
    def __init__(self, corpus: Corpus):
        super(TFKeywords, self).__init__(corpus)
        if type(corpus) != SpacyCorpus:
            raise TypeError(f"TFKeywords requires {SpacyCorpus.__name__}. "
                            f"Please process it with {SpacyProcessor.__name__}")

        self._vocab = corpus.vocab
        self._threshold = None
        self._normalise = True
        self._log = False

    def freq_threshold(self, threshold: int):
        self._threshold = threshold
        return self

    def normalise(self, to_normalise=True):
        """ Normalise by the number of words in the corpus. """
        self._normalise = to_normalise
        return self

    def log_freqs(self, to_log=True):
        """ Log the score from term frequencies. (Zip's law) """
        self._log = to_log
        return self

    def set_df_range(self, min_, max_):
        """ """
        pass

    def extracted(self):
        word_freqs = self._count(self.corpus, normalise=self._normalise, log=self._log)
        return word_freqs

    def _count(self, corpus: Corpus, normalise: bool, log: bool):
        doc_freq_counter = Counter()
        freq_counter = Counter()
        threshold_diff_to_adjust = 0

        _no_puncs_no_stopwords = no_puncs_no_stopwords(self._vocab)
        for d in corpus.texts():
            per_doc_freqs = dict()
            for _, start, end in _no_puncs_no_stopwords(d):
                t = d[start:end].text.lower()
                current = per_doc_freqs.get(t, 0)
                per_doc_freqs[t] = current

            # apply threshold here and count the difference.
            for k, v in per_doc_freqs.items():
                _orig_value = v
                per_doc_freqs[k] = max(v, self._threshold)
                threshold_diff_to_adjust += _orig_value - self._threshold
            freq_counter.update(per_doc_freqs)
            # set a max on per_doc_freqs to 1 and add to doc_freq_counter
            doc_freq_counter.update({k: 1 for k, _ in per_doc_freqs.items()})

        freq_counter = dict(freq_counter)
        num_words = corpus.num_words - threshold_diff_to_adjust
        if log:
            for k in freq_counter.keys():
                freq_counter[k] = math.log(freq_counter.get(k))
            num_words = math.log(num_words)
        if normalise:
            for k in freq_counter.keys():
                freq_counter[k] = (freq_counter.get(k) / num_words) * 100
        return sorted(freq_counter.items(), key=lambda kv: kv[1], reverse=True)


if __name__ == '__main__':
    from juxtorpus.corpus import Corpus, CorpusBuilder
    import re

    tweet_wrapper = re.compile(r'([ ]?<[/]?TWEET>[ ]?)')

    builder = CorpusBuilder('/Users/hcha9747/Downloads/Geolocated_places_climate_with_LGA_and_remoteness_with_text.csv')
    builder.set_text_column('text')
    builder.set_nrows(100)
    builder.set_preprocessors([lambda text: tweet_wrapper.sub('', text)])
    corpus = builder.build()

    from juxtorpus.corpus.processors import SpacyProcessor
    import spacy

    nlp = spacy.load('en_core_web_sm')
    spacy_processor = SpacyProcessor(nlp)
    corpus = spacy_processor.run(corpus)

    from juxtorpus.features.keywords import TFKeywords

    tf = TFKeywords(corpus)
    tf.freq_threshold(3).normalise().log_freqs(False)
    print('\n'.join((str(x) for x in tf.extracted()[:10])))
