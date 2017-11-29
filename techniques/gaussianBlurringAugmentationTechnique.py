from itechnique import ITechnique
import cv2
import numpy as np

class gausianBlurringAugmentationTechnique(ITechnique):

    # Valid values for kernel are 3,5,7,9, and 11
    def __init__(self,kernel=3):
        ITechnique.__init__(self,False)
        if (not (kernel in [3,5,7,9,11])):
            raise NameError("Invalid value for kernel")
        self.kernel = kernel

    def apply(self, image):
        blurred = cv2.GaussianBlur(image, (self.kernel, self.kernel),0)
        return blurred