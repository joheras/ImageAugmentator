from itechnique import ITechnique
import random

class dropoutAugmentationTechnique(ITechnique):

    # percentage of pixels to dropout is a value between 0 and 1
    def __init__(self,percentage=0.05):
        ITechnique.__init__(self,False)
        self.percentage = percentage

    def apply(self, image):
        channels = len(image.shape)
        elems =[(x, y) for x in range(0, image.shape[0]) for y in range(0, image.shape[1])]
        random.shuffle(elems)
        dropoutelems = elems[0:int(self.percentage*len(elems))]
        imageC = image.copy()
        for dropoutelem in dropoutelems:
            if channels==3:
                imageC[dropoutelem[0],dropoutelem[1]] = [0,0,0]
            else:
                imageC[dropoutelem[0], dropoutelem[1]] = 0
        return imageC

# technique = dropoutAugmentationTechnique(0.5)
# image = cv2.imread("LPR1.jpg")
# cv2.imshow("t",technique.applyForClassification(image))
# cv2.waitKey(0)