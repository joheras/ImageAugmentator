from itechnique import ITechnique
import cv2
import numpy as np

class blurringAugmentationTechnique(ITechnique):

    # Valid values for kernel are 3,5,7,9, and 11
    def __init__(self,ksize=3):
        ITechnique.__init__(self,False)
        if (not (ksize in [3,5,7,9,11])):
            raise NameError("Invalid value for kernel")
        self.ksize = ksize

    def apply(self, image):
        blurred = cv2.blur(image, (self.ksize,self.ksize))
        return blurred


# technique = blurringAugmentationTechnique(5)
# image = cv2.imread("LPR1.jpg")
# cv2.imshow("resized",technique.apply(image))
# cv2.waitKey(0)