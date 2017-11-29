from itechnique import ITechnique
import cv2
import numpy as np

class saltAndPepperNoiseAugmentationTechnique(ITechnique):


    def __init__(self,low=0,up=25):
        ITechnique.__init__(self,False)
        self.low = low
        self.up=up

    def apply(self, image):
        im = np.zeros(image.shape, np.uint8)
        if(len(image.shape)!=3):
            l = self.low
            u = self.up
        else:
            l = (self.low,self.low,self.low)
            u = (self.up,self.up,self.up)
        cv2.randu(im, l,u)
        image_noise = cv2.add(image, im)
        return image_noise


# image = cv2.imread("LPR1.jpg")
# print (image.shape)
# im = np.zeros(image.shape, np.uint8)
# m = (50,50,50)
# s = (10,10,10)
# cv2.randu(im, (0,0,0),(50,50,50))
# image_noise=cv2.add(image, im)
# cv2.imshow("original",image)
# cv2.imshow("noise",image_noise)
# cv2.waitKey(0)