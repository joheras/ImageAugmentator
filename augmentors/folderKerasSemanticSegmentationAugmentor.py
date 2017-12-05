from mahotas.demos import image_path

from iaugmentor import IAugmentor
from imutils import paths
import os
import cv2
from sklearn.externals.joblib import Parallel, delayed
import random
import numpy as np

def readAndGenerateImageSegmentation(image,label,generators):
    newimage = image
    newlabel = label
    for (j, generator) in enumerate(generators):
        if (random.randint(0, 100) > 50):
            newimage = generator.applyForSegmentation(newimage,newlabel)

    return (newimage,newlabel)

# This class serves to generate images for a semantic segmentation
# problem where all the images are organized in two folders called
# images and labels. Both folders must contain the same number of images
# and with the same names (but maybe different extensions). Example
# - Folder
# |- images
#    |- image1.jpg
#    |- image2.jpg
#    |- ...
# |- labels
#    |- image1.tiff
#    |- image2.tiff
#    |- ...
# where Folder/labels/image1.tiff is the annotation of the image Folder/images/image1.jpg.
# Hence, both images must have the same size.
class FolderKerasSemanticSegmentationAugmentor:

    def __init__(self,inputPath,labelsExtension=".tiff",batchSize=32):
        IAugmentor.__init__(self)
        self.inputPath = inputPath
        self.imagesPath = inputPath+"images/"
        self.labelsPath = inputPath + "labels/"
        # output path represents the folder where the images will be stored
        self.generators = []
        self.labelsExtension=labelsExtension
        self.readImagesAndAnnotations()
        self.batchSize = batchSize

    def addGenerator(self, generator):
        self.generators.append(generator)

    def readImagesAndAnnotations(self):
        self.imagePaths = list(paths.list_files(self.imagesPath,validExts=(".jpg", ".jpeg", ".png", ".bmp",".tiff",".tif")))
        self.labelPaths = list(paths.list_files(self.labelsPath,validExts=(".jpg", ".jpeg", ".png", ".bmp",".tiff",".tif")))
        self.numImages = len(self.imagePaths)
        if (len(self.imagePaths)!=len(self.labelPaths)):
            raise Exception("The number of files is different in the folder of images and in the folder of labels")



    def applyAugmentation(self,passes=np.inf):
        epochs = 0
        while epochs < passes:

            for i in np.arange(0, self.numImages, self.batchSize):
                imagPaths = self.imagePaths[i:i+self.batchSize]
                labPaths = self.labelPaths[i:i+self.batchSize]
                images = [cv2.imread(imagePath) for imagePath in imagPaths]
                labels = [cv2.imread(labelPath) for labelPath in labPaths]
                images_labels = [readAndGenerateImageSegmentation(image,label,self.generators) for (image,label) in zip(images,labels)]
                images = [i[0] for i in images_labels]
                labels = [i[1] for i in images_labels]
                yield (images,labels)


            epochs += 1


