from itechnique import ITechnique
import cv2
import numpy as np

class raiseBlueAugmentationTechnique(ITechnique):

    # Valid values for pover are in the range (0.25,4]
    def __init__(self,power=0.9):
        ITechnique.__init__(self,False)
        if (power<=0.25 or power >4):
            raise NameError("Invalid value for power")
        self.power = power

    def apply(self, image):
        if(len(image.shape)!=3):
            raise NameError("Not applicable technique")
        identityG = np.arange(256, dtype=np.dtype('uint8'))
        identityB = np.array([((i / 255.0) ** self.power) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        identityR = np.arange(256, dtype=np.dtype('uint8'))
        lut = np.dstack((identityB, identityG, identityR))

        # apply gamma correction using the lookup table
        image = cv2.LUT(image, lut)
        return image