from itechnique import ITechnique
import cv2
import numpy as np

class shearingAugmentationTechnique(ITechnique):

    def __init__(self,a=0.5):
        ITechnique.__init__(self,True)
        self.a = a

    def __shear(self,image, a):
        # define the translation matrix and perform the translation
        M = np.float32([[1, a, 0], [0, 1, 0]])
        sheared = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

        # return the translated image
        return sheared

    def apply(self, image):
        sheared = self.__shear(image, self.a)
        return sheared

# technique = shearingAugmentationTechnique(0.5)
# image = cv2.imread("LPR1.jpg")
# cv2.imshow("t",technique.applyForClassification(image))
# cv2.waitKey(0)
