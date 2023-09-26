from abc import ABCMeta


class TopicModel(metaclass=ABCMeta):
    pass


from .topsbm import TopSBM

__all__ = ['TopSBM']
