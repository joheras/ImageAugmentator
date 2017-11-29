from itechnique import ITechnique
import cv2
import numpy as np

class gaussianNoiseAugmentationTechnique(ITechnique):


    def __init__(self,mean=0,sigma=10):
        ITechnique.__init__(self,False)
        self.mean = mean
        self.sigma=sigma

    def apply(self, image):
        im = np.zeros(image.shape, np.uint8)
        if(len(image.shape)==2):
            m = self.mean
            s = self.sigma
        else:
            m = (self.mean,self.mean,self.mean)
            s = (self.sigma,self.sigma,self.sigma)
        cv2.randn(im, m, s)
        image_noise = cv2.add(image, im)
        return image_noise



