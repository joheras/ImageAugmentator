from abc import ABCMeta, abstractmethod

class ITechnique:
    __metaclass__ = ABCMeta

    def __init__(self, changeLabel=False):
        self.changeLabel = changeLabel

    @abstractmethod
    def apply(self, image):
        raise NotImplementedError