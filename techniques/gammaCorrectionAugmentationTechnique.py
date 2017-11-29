from itechnique import ITechnique
import cv2
import numpy as np

class gammaCorrectionAugmentationTechnique(ITechnique):

    # Valid values for gamma are in the range (0,2.5]
    def __init__(self,gamma=1.5):
        ITechnique.__init__(self,False)
        if (gamma<=0 or gamma >2.5):
            raise NameError("Invalid value for gamma")
        self.gamma = gamma

    def apply(self, image):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)