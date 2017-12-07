from iaugmentor import IAugmentor
from imutils import paths
import os
import cv2
from sklearn.externals.joblib import Parallel, delayed
import xml.etree.ElementTree as ET
from utils import prettify
import random
import numpy as np


#
def readAndGenerateImage(generators, i_and_imagePath):
    (i, imagePath) = i_and_imagePath
    image = cv2.imread(imagePath)
    name = imagePath.split(os.path.sep)[-1]
    labelPath = '/'.join(imagePath.split(os.path.sep)[:-1]) + "/" + name[0:name.rfind(".")] + ".xml"
    tree = ET.parse(labelPath)
    root = tree.getroot()
    objects = root.findall('object')
    if (len(objects) > 1):
        raise Exception("The xml should contain at least one object")
    boxes = []
    for object in objects:
        category = object.find('name').text
        bndbox = object.find('bndbox')
        x = int(bndbox.find('xmin').text)
        y = int(bndbox.find('ymin').text)
        w = int(bndbox.find('ymax').text) - y
        h = int(bndbox.find('xmax').text) - x
        boxes.append((category, (x, y, w, h)))

    newimage=image
    for (j, generator) in enumerate(generators):
        if (random.randint(0, 100) > 50):
            (newimage, boxes) = generator.applyForDetection(image, boxes)
    return (newimage, box)




    #
# # This class serves to generate images for a localization
# # problem where all the images in the given folder, and the labels
# # are given in the same folder with the same name and using the PASCAL VOC format.
# # Example:
# # - Folder
# # |- image1.jpg
# # |- image1.xml
# # |- image2.jpg
# # |- image2.xml
# # |- ...
# #
#
class PascalVOCKerasDetectionAugmentor:

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
        self.numImages = len(self.imagePaths)
        self.labelPaths = list(paths.list_files(self.inputPath,validExts=(".xml")))
        if (len(self.imagePaths) != len(self.labelPaths)):
            raise Exception("The number of images is different to the number of annotations")

    def applyAugmentation(self,passes=np.inf):

        epochs = 0
        while epochs < passes:

            for i in np.arange(0, self.numImages, self.batchSize):
                imagPaths = self.imagePaths[i:i + self.batchSize]
                images_labels = [readAndGenerateImage(self.generators,x) for x in enumerate(imagPaths)]
                images = [i[0] for i in images_labels]
                labels = [i[1] for i in images_labels]
                yield (images, labels)

            epochs += 1