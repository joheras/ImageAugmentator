from itechnique import ITechnique
import cv2
import numpy as np

class bilateralBlurringAugmentationTechnique(ITechnique):

    # Examples for values of diameter, sigmaColor, and sigmaSpace are
    # (11,21,7), (11,41,21), (11,61,39).
    def __init__(self,diameter=11,sigmaColor=21,sigmaSpace=7):
        ITechnique.__init__(self,False)
        self.diameter = diameter
        self.sigmaColor = sigmaColor
        self.sigmaSpace = sigmaSpace

    def apply(self, image):
        blurred = cv2.bilateralFilter(image, self.diameter, self.sigmaColor, self.sigmaSpace)
        return blurred