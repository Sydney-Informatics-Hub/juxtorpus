from abc import ABCMeta, abstractmethod
import pandas as pd
from typing import Generator

from juxtorpus.interfaces.clonable import Clonable, MASK


class Meta(Clonable, metaclass=ABCMeta):
    def __init__(self, id_: str):
        self._id = id_

    @property
    def id(self):
        return self._id

    @abstractmethod
    def apply(self, func) -> pd.Series:
        raise NotImplementedError()

    @abstractmethod
    def groupby(self, grouper) -> Generator[tuple[str, MASK], None, None]:
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError()

    @abstractmethod
    def cloned(self, texts: 'pd.Series[str]', mask: MASK):
        raise NotImplementedError()

    @abstractmethod
    def head(self, n: int):
        raise NotImplementedError()

    @abstractmethod
    def summary(self) -> pd.DataFrame:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} [Id: {self.id}]>"
