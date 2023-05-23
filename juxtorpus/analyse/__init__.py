from juxtorpus.corpus import Corpus

# todo: import the analyse functions from each submodule.
from .topic_model import analyse_with_lda
from .sentiment import analyse_with_sentiment

_methods = {
    'topic_model': analyse_with_lda,
    'sentiment': analyse_with_sentiment
}


def analyse(corpus: Corpus, method: str, add_results: bool, **kwargs):
    func = _methods.get(method, None)
    if func is None: raise ValueError(f"{method} is not one of {','.join(_methods.keys())}")

    results_prefix = '#' + method
    return func(corpus, add_results=add_results, results_prefix=results_prefix, **kwargs)


__all__ = ['analyse']
