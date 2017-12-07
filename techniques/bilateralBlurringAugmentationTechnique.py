from itechnique import ITechnique
import cv2
import numpy as np

class bilateralBlurringAugmentationTechnique(ITechnique):

    # Examples for values of diameter, sigmaColor, and sigmaSpace are
    # (11,21,7), (11,41,21), (11,61,39).
    def __init__(self,parameters):
        ITechnique.__init__(self,False)
        if parameters["diameter"]:
            self.diameter = parameters["diameter"]
        else:
            self.diameter = 11
        if parameters["sigmaColor"]:
            self.sigmaColor = parameters["sigmaColor"]
        else:
            self.sigmaColor = 21
        if parameters["sigmaSpace"]:
            self.sigmaSpace = parameters["sigmaSpace"]
        else:
            self.sigmaSpace = 7



    def apply(self, image):
        blurred = cv2.bilateralFilter(image, self.diameter, self.sigmaColor, self.sigmaSpace)
        return blurred