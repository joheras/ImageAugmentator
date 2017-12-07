from itechnique import ITechnique
import cv2
import numpy as np

class raiseHueAugmentationTechnique(ITechnique):

    # Valid values for pover are in the range (0.25,4]
    def __init__(self,parameters):
        ITechnique.__init__(self,False)
        if parameters["power"]:
            self.power = parameters["power"]
        else:
            self.power = 1.5

        if (self.power<=0.25 or self.power >4):
            raise NameError("Invalid value for power")

    def apply(self, image):
        if(len(image.shape)!=3):
            raise NameError("Not applicable technique")
        imageHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        identityV = np.arange(256, dtype=np.dtype('uint8'))
        identityH = np.array([((i / 255.0) ** self.power) * 255
                              for i in np.arange(0, 256)]).astype("uint8")
        identityS = np.arange(256, dtype=np.dtype('uint8'))
        lut = np.dstack((identityH, identityS, identityV))

        # apply gamma correction using the lookup table
        imageHSV = cv2.LUT(imageHSV, lut)
        imageRGB = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2BGR)
        return imageRGB