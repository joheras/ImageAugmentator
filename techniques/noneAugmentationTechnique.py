from itechnique import ITechnique
import cv2

class noneAugmentationTechnique(ITechnique):

    # Valid values for kernel are 3,5,7,9, and 11
    def __init__(self):
        ITechnique.__init__(self,False)


    def apply(self, image):
        return image