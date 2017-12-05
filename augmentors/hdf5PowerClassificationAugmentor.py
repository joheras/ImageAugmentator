from iaugmentor import IAugmentor
from sklearn.preprocessing import LabelEncoder
from imutils import paths
import os
import cv2
from sklearn.externals.joblib import Parallel, delayed
from utils.aspectawarepreprocessor import AspectAwarePreprocessor
from utils.hdf5datasetwriter import HDF5DatasetWriterClassification
import progressbar

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
class HDF5PowerClassificationAugmentor:

    # All images must have same width and height
    def __init__(self,inputPath,outputPath,width,height):
        IAugmentor.__init__(self)
        self.inputPath = inputPath
        # output path represents the h5py file where dataset will be stored
        self.outputPath = outputPath
        self.generators = []
        self.width = width
        self.height = height
        self.aw = AspectAwarePreprocessor(width,height)

    def addGenerator(self, generator):
        self.generators.append(generator)

    def readImagesAndAnnotations(self):
        self.imagePaths = list(paths.list_images(self.inputPath))


    def applyAugmentation(self):
        self.readImagesAndAnnotations()
        le = LabelEncoder()
        labels = [p.split(os.path.sep)[-2] for p in self.imagePaths]
        labels = le.fit_transform(labels)
        writer = HDF5DatasetWriterClassification((len(self.imagePaths)*(2**(len(self.generators)-1)),self.width,self.height,3),
                                   self.outputPath)
        # We need to define this function outside to work in parallel.
        writer.storeClassLabels(le.classes_)
        widgets = ["Processing images: ", progressbar.Percentage(), " ",
                   progressbar.Bar(), " ", progressbar.ETA()]
        pbar = progressbar.ProgressBar(maxval=len(self.imagePaths),
                                       widgets=widgets).start()
        for i_and_imagePath in enumerate(zip(self.imagePaths,labels)):
            (i, (imagePath,label)) = i_and_imagePath
            image = cv2.imread(imagePath)
            image = self.aw.preprocess(image)

            images = [image]

            for (j, generator) in enumerate(self.generators):
                newimages = []
                for (k,im) in enumerate(images):
                    newimage = generator.applyForClassification(im)
                    newimage = self.aw.preprocess(newimage)
                    writer.add([newimage], [label])
                    newimages.append(newimage)
                images = newimages

            pbar.update(i)
        writer.close()
        pbar.finish()

# # Example
# augmentor = HDF5PowerClassificationAugmentor(
#     "/home/joheras/datasets/cats_and_dogs_small/train/",
#     "/home/joheras/datasets/cats_and_dogs_small/database.hdf5",224,224
# )

# from techniques.averageBlurringAugmentationTechnique import averageBlurringAugmentationTechnique
# from techniques.bilateralBlurringAugmentationTechnique import bilateralBlurringAugmentationTechnique
# from techniques.gaussianNoiseAugmentationTechnique import gaussianNoiseAugmentationTechnique
# from techniques.rotateAugmentationTechnique import rotateAugmentationTechnique
# from techniques.flipAugmentationTechnique import flipAugmentationTechnique
# from techniques.noneAugmentationTechnique import noneAugmentationTechnique
# from generator import Generator
# import time
# augmentor.addGenerator(Generator(noneAugmentationTechnique()))
# augmentor.addGenerator(Generator(averageBlurringAugmentationTechnique()))
# augmentor.addGenerator(Generator(bilateralBlurringAugmentationTechnique()))
# augmentor.addGenerator(Generator(gaussianNoiseAugmentationTechnique()))
# augmentor.addGenerator(Generator(rotateAugmentationTechnique()))
# augmentor.addGenerator(Generator(flipAugmentationTechnique()))
# start = time.time()
# augmentor.applyAugmentation()
# end = time.time()
# print(end - start)