from itechnique import ITechnique
import cv2
import numpy as np

class blurringAugmentationTechnique(ITechnique):

    # Valid values for ksize are 3,5,7,9, and 11
    def __init__(self,parameters):
        ITechnique.__init__(self,False)
        if 'ksize' in parameters.keys():
            self.ksize = int(parameters["ksize"])
        else:
            self.ksize = 3
        if (not (self.ksize in [3,5,7,9,11])):
            raise NameError("Invalid value for ksize")


    def apply(self, image):
        blurred = cv2.blur(image, (self.ksize,self.ksize))
        return blurred


# technique = blurringAugmentationTechnique(5)
# image = cv2.imread("LPR1.jpg")
# cv2.imshow("resized",technique.apply(image))
# cv2.waitKey(0)