from iaugmentor import IAugmentor
from imutils import paths
import os
import cv2
import numpy as np
import random


def readAndGenerateImage(image, generators):
    newimage = image
    for (j, generator) in enumerate(generators):
        if (random.randint(0,100)>50):
            newimage = generator.applyForClassification(newimage)

    return newimage

# This class serves to generate images for a classification
# problem where all the images are organized by folders
# distributed by labels. Example:
# - Folder
# |- cats
#    |- image1.jpg
#    |- image2.jpg
#    |- ...
# |- dogs
#    |- image1.jpg
#    |- image2.jpg
#    |- ...
class FolderKerasLinearClassificationAugmentor:

    def __init__(self,inputPath,parameters):
        IAugmentor.__init__(self)
        self.inputPath = inputPath
        # output path represents the folder where the images will be stored
        if parameters["batchSize"]:
            self.batchSize = parameters["batchSize"]
        else:
            self.batchSize = 32
        self.generators = []
        self.readImagesAndAnnotations()

    def addGenerator(self, generator):
        self.generators.append(generator)

    def readImagesAndAnnotations(self):
        self.imagePaths = list(paths.list_images(self.inputPath))
        random.shuffle(self.imagePaths)
        self.numImages = len(self.imagePaths)
        self.labels = [p.split(os.path.sep)[-2] for p in self.imagePaths]



    def applyAugmentation(self,passes=np.inf):
        epochs = 0
        while epochs < passes:

            for i in np.arange(0, self.numImages, self.batchSize):
                imagPaths = self.imagePaths[i:i+self.batchSize]
                labels = self.labels[i:i+self.batchSize]
                images = [cv2.imread(imagePath) for imagePath in imagPaths]
                images = [readAndGenerateImage(image,self.generators) for image in images]

                yield (images,labels)


            epochs += 1









