from abc import ABC
from typing import Callable, Optional, Union, Iterable, Collection
import pandas as pd
from ipywidgets import HTML, Label
from IPython.display import display

from juxtorpus.corpus import Corpus
from juxtorpus.interfaces import Container
from juxtorpus.viz import Widget, Viz
from juxtorpus.viz.widgets import CorporaWidget


class Corpora(Container, Widget, Viz, ABC):
    """ Corpora
    This is a container class that acts as a registry of corpus.
    """

    def __init__(self, list_of_corpus: Optional[list[Corpus]] = None):
        self._map = {id(c): c for c in list_of_corpus} if list_of_corpus else dict()

    def _find_corpus(self, name: str) -> Optional[Corpus]:
        """ Return the corpus in this container based on name. """
        # todo: if this is too slow, cache a sorted list of corpus names and use binary tree search.
        #  the cache should be updated each time a corpus is added.
        for c in self._map.values():
            if c.name == name: return c
        return None

    def get(self, name: str) -> Optional[Corpus]:
        """ Returns the Corpus with name. Returns None if not found. """
        return self._find_corpus(name)

    def __getitem__(self, name: str) -> Optional[Corpus]:
        return self.get(name)

    def add(self, corpus: Union[Corpus, Iterable[Corpus]]):
        if isinstance(corpus, Corpus):
            self._map[id(corpus)] = corpus

        if type(corpus) in (list, set):
            for c in corpus: self.add(c)

    def remove(self, name: str) -> bool:
        """ Remove the corpus given name. Returns False if not exist."""
        # todo: maybe this should raise an exception instead.
        try:
            del self._map[name]
            return True
        except KeyError as ke:
            return False

    def clear(self):
        """ Clear all corpus from corpora. """
        self._map = dict()

    def items(self) -> list[str]:
        """ List all the corpus names in the corpora. """
        return [c.name for c in self._map.values()]

    def widget(self):
        """ Returns a dashboard of existing corpus """
        return CorporaWidget(self).widget()

    def render(self):
        """ Visualise all the corpus currently contained within the Corpora. """
        if len(self) <= 0:
            display(Label('There is currently no Corpus contained in this Corpora.'))
        else:
            table_data = []
            for corpus in self._map.values():
                table_data.append([
                    corpus.name,
                    corpus.parent.name if corpus.parent else '',
                    corpus.__class__.__name__,
                    len(corpus),
                    len(corpus.dtm.vocab(nonzero=True)),
                    ', '.join(corpus.meta.keys())
                ])
            table_df = pd.DataFrame(table_data, columns=['Corpus', 'Parent', 'Type', 'Docs', 'Vocab', 'Metas'])
            table_widget = HTML(table_df.to_html(index=False))
            display(table_widget)

    def __len__(self):
        return len(self._map)
