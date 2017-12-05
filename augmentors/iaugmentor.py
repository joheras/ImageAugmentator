from abc import ABCMeta,abstractmethod

class IAugmentor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def addGenerator(self, generator):
        None

    @abstractmethod
    def readImagesAndAnnotations(self):
        raise NotImplementedError

    @abstractmethod
    def applyAugmentation(self):
        raise NotImplementedError
