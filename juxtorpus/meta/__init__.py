from abc import ABCMeta, abstractmethod
from typing import List, Callable, Any, Set, Union, Iterable
import pandas as pd
from spacy.tokens import Doc
import pathlib

""" 
A Collection of data classes representing Corpus Metadata.
"""


class LazyLoader(metaclass=ABCMeta):
    @abstractmethod
    def load(self) -> Iterable:
        """"""
        raise NotImplementedError()


class LazySeries(LazyLoader):
    def __init__(self, paths: list[pathlib.Path], nrows: int, pd_read_func):
        """
        :param paths: paths of the csv
        :param nrows: max number of rows
        :param pd_read_func: One of pandas read functions (i.e. dataframe constructors)
        """
        self._paths = paths if isinstance(paths, list) else list(paths)
        self._nrows = nrows
        self._read_func = pd_read_func

    @property
    def nrows(self):
        return self._nrows

    @property
    def paths(self):
        return self._paths

    def load(self):
        return pd.concat(self._yield_series(), axis=0)

    def _yield_series(self) -> pd.Series:
        # load all
        if self._nrows is None:
            for path in self._paths:
                yield self._read_func(path).squeeze("columns")
        # load up to nrows
        else:
            current = 0
            for path in self.paths:
                series: pd.Series = self._read_func(path, nrows=self._nrows - current).squeeze("columns")
                current += len(series)
                if current <= self._nrows:
                    yield series


class LazySpacyPipe(LazyLoader):
    def __init__(self, texts: Iterable):
        self.texts = texts

    def load(self) -> Iterable[Doc]:
        return nlp.pipe(self.texts)


class Meta(metaclass=ABCMeta):
    def __init__(self, id_: str, df_col: str):
        self._id = id_
        self._df_col = df_col

    @property
    def id(self):
        return self._id

    @abstractmethod
    def mask_on_condition(self, cond: Callable[[Any], bool]) -> 'pd.Series[bool]':
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [Id: {self.id}]>"


class ItemMasker(metaclass=ABCMeta):

    @abstractmethod
    def mask_on_items(self, items: Set[str], op: str) -> 'pd.Series[bool]':
        op = op.upper()
        if op not in (self.OP_OR, self.OP_AND):
            raise ValueError(f"{op} is not supported. Must be either OR or AND.")
        if op == self.OP_OR:
            return self._or_mask(items)
        else:
            return self._and_mask(items)

    @property
    def OP_OR(self):
        return 'OR'

    @property
    def OP_AND(self):
        return 'AND'

    @abstractmethod
    def _or_mask(self, items: Set[str]):
        """ Return the boolean mask where True means 'any' item in the set matches. """
        raise NotImplementedError()

    @abstractmethod
    def _and_mask(self, items: Set[str]):
        """ Return the boolean mask where True means 'all' item in the set matches. """
        raise NotImplementedError()


""" Metadata of pandas series may be given OR derived. """


class SeriesMeta(Meta, metaclass=ABCMeta):
    def __init__(self, id_: str, series: Union[LazySeries, pd.Series]):
        super(SeriesMeta, self).__init__(id_, id_)
        self._dtype = None
        self._series = series

    @property
    def series(self) -> pd.Series:
        if isinstance(self._series, LazySeries):
            self._series = self._series.load()
        return self._series

    def mask_on_condition(self, cond: Callable[[Any], bool]) -> 'pd.Series[bool]':
        return self.series.apply(lambda x: cond(x))

    def __repr__(self):
        return f"{super(SeriesMeta, self).__repr__()[:-2]}, DType: {self._dtype}]"


class CategoricalSeriesMeta(SeriesMeta, ItemMasker):
    def mask_on_items(self, items: Set[str], op: str) -> 'pd.Series[bool]':
        return super(CategoricalSeriesMeta, self).mask_on_items(items, op)

    def _series_to_filter_on(self) -> pd.Series:
        return self.series

    def _or_mask(self, items: Set[str]):
        return self._series_to_filter_on().isin(items)

    def _and_mask(self, items: Set[str]):
        init_series = pd.Series((False for _ in range(len(self._series_to_filter_on()))))
        for i, item in enumerate(items):
            if i == 0:
                init_series = init_series | self.series.apply(lambda x: x == item)
            else:
                init_series = init_series & self.series.apply(lambda x: x == item)
        return init_series


class DelimitedStrSeriesMeta(CategoricalSeriesMeta):
    """ String delimited meta data representing items. """

    def __init__(self, delimiter: str, *args, **kwargs):
        super(DelimitedStrSeriesMeta, self).__init__(*args, **kwargs)
        self.delimiter = delimiter

    def _series_to_filter_on(self) -> pd.Series:
        return self.series.apply(lambda x: x.split(self.delimiter))


""" Metadata from spaCy docs can only be derived metadata. """


class DocMeta(Meta):
    """ This class represents the metadata stored within the spacy Docs """

    def __init__(self, id_: str, df_col: str, attr: str, doc_generator: Iterable[Doc]):
        super(DocMeta, self).__init__(id_, df_col)
        self._attr = attr
        self._doc_generator = doc_generator

    @property
    def attribute(self):
        return self._attr

    def mask_on_condition(self, cond: Callable[[Any], bool]) -> 'pd.Series[bool]':
        """ Return a boolean mask based on condition applied to the metadata in the doc."""
        return pd.Series((cond(doc.get_extension(self._attr)) for doc in self._doc_generator))

    def _get_doc_attr(self, doc: Doc) -> Any:
        """ Returns a built-in spacy entity OR a custom entity. """
        return doc.get_extension(self._attr) if doc.has_extension(self._attr) else getattr(doc, self._attr)

    def __repr__(self):
        return f"{super(DocMeta, self).__repr__()[:-2]}, Attr: {self._attr}]"


class DocItemMeta(DocMeta, ItemMasker):

    def __init__(self, *args, **kwargs):
        super(DocItemMeta, self).__init__(*args, **kwargs)

    def mask_on_items(self, items: Set[str], op: str) -> 'pd.Series[bool]':
        return super(DocItemMeta, self).mask_on_items(items, op)

    def _or_mask(self, items: Set[str]):
        return pd.Series((doc.get_extension(self._attr) in items for doc in self._doc_generator))

    def _and_mask(self, items: Set[str]):
        _mask = list()
        for doc in self._doc_generator:
            doc_item = doc.get_extension(self._attr)
            _mask.append(doc_item in items)
        return pd.Series(_mask)


class DocItemsMeta(DocItemMeta, ItemMasker):

    def __init__(self, *args, **kwargs):
        super(DocItemMeta, self).__init__(*args, **kwargs)

    def _or_mask(self, items: Set[str]):
        _mask = list()
        for doc in self._doc_generator:
            doc_items = self._get_doc_attr(doc)
            if not isinstance(doc_items, set): doc_items = set((doc_item.text for doc_item in doc_items))
            _mask.append(len(doc_items.intersection(items)) > 0)  # items must be a set.
        return pd.Series(_mask)

    def _and_mask(self, items: Set[str]):
        _mask = list()
        for doc in self._doc_generator:
            doc_items = self._get_doc_attr(doc)
            if not isinstance(doc_items, set): doc_items = set((doc_item.text for doc_item in doc_items))
            _mask.append(doc_items.intersection(items) == len(doc_items))
        return pd.Series(_mask)
