from itechnique import ITechnique
import cv2
import numpy as np

class equalizeHistogramAugmentationTechnique(ITechnique):

    # Valid values for kernel are 3,5,7,9, and 11
    def __init__(self):
        ITechnique.__init__(self,False)

    def apply(self, image):
        if len(image.shape) == 3:
            b, g, r = cv2.split(image)
            red = cv2.equalizeHist(r)
            green = cv2.equalizeHist(g)
            blue = cv2.equalizeHist(b)
            return cv2.merge((blue, green, red))
        else:
            return cv2.equalizeHist(image)

technique = equalizeHistogramAugmentationTechnique()
image = cv2.imread("LPR1.jpg")
cv2.imshow("resized",technique.apply(image))
cv2.waitKey(0)