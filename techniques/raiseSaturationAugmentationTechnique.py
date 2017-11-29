from itechnique import ITechnique
import cv2
import numpy as np

class raiseSaturationAugmentationTechnique(ITechnique):

    # Valid values for pover are in the range (0.25,4]
    def __init__(self,power=1.5):
        ITechnique.__init__(self,False)
        if (power<=0.25 or power >4):
            raise NameError("Invalid value for power")
        self.power = power

    def apply(self, image):
        if(len(image.shape)!=3):
            raise NameError("Not applicable technique")
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        identityH = np.arange(256, dtype=np.dtype('uint8'))
        identityS = np.array([((i / 255.0) ** self.power) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        identityV = np.arange(256, dtype=np.dtype('uint8'))
        lut = np.dstack((identityH, identityS, identityV))

        # apply gamma correction using the lookup table
        imageHSV=cv2.LUT(imageHSV, lut)
        imageRGB = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)
        return imageRGB