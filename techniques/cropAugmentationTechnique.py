from itechnique import ITechnique
import cv2

class cropAugmentationTechnique(ITechnique):

    # percentage is a value between 0 and 1
    # startFrom indicates the starting point of the cropping,
    # the possible values are TOPLEFT, TOPRIGHT, BOTTOMLEFT,
    # BOTTOMRIGHT, and CENTER
    def __init__(self,percentage=0.9,startFrom='TOPLEFT'):
        ITechnique.__init__(self,True)
        if (percentage <0 or percentage >1):
            raise NameError("Invalid value for cropping")
        if startFrom not in ['TOPLEFT', 'TOPRIGHT', 'BOTTOMLEFT', 'BOTTOMRIGHT', 'CENTER']:
            raise NameError("Invalid value for cropping")
        self.percentage = percentage
        self.startFrom = startFrom

    def apply(self, image):
        (w,h) = image.shape[:2]
        newW = int(w*self.percentage)
        newH = int(h * self.percentage)
        if self.startFrom == 'TOPLEFT':
            crop = image[0:newW,0:newH]
        if self.startFrom == 'BOTTOMLEFT':
            crop = image[w-newW:w, 0:newH]
        if self.startFrom == 'TOPRIGHT':
            crop = image[0:newW,h-newH:h]
        if self.startFrom == 'BOTTOMRIGHT':
            crop = image[w-newW:w, h - newH:h]
        if self.startFrom == 'CENTER':
            crop = image[int(w/2 - newW/2):int(w/2 + newW/2), int(h/2 - newH/2):int(h/2 + newH/2)]
        return crop


# Example
# technique = cropAugmentationTechnique(0.5,'TOPRIGHT')
# image = cv2.imread("LPR1.jpg")
# cv2.imshow("original",image)
# cv2.imshow("new",technique.apply(image))
# cv2.waitKey(0)