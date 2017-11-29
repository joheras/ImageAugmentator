from iaugmentor import IAugmentor
from imutils import paths
import os
import cv2
from sklearn.externals.joblib import Parallel, delayed
import xml.etree.ElementTree as ET



tree = ET.parse('/home/joheras/datasets/estomas/307A5_Multifocus Image(1).xml')
root = tree.getroot()
objects= root.findall('object')
print(objects[0].find('name').text)



#
def readAndGenerateImage(outputPath, generators, i_and_imagePath):
    (i, imagePath) = i_and_imagePath
    image = cv2.imread(imagePath)
    name = imagePath.split(os.path.sep)[-1]
    labelPath = '/'.join(imagePath.split(os.path.sep)[:-1]) + name[0:name.rfind(".")] + ".xml"
    tree = ET.parse(labelPath)
    root = tree.getroot()
    objects = root.findall('object')
    if(len(objects)!=1):
        raise Exception("Since this is a localization problem, the xml should only contain one object")
    object = objects[0]
    category = object.find('name').text
    x  = int(object.find('xmin').text)
    y = int(object.find('ymin').text)
    w = int(object.find('ymax').text)-y
    h = int(object.find('xmax').text) - x

    # HEREEEEEEEEE, we need to create the xml
    for (j, generator) in enumerate(generators):
        (newimage,box) = generator.applyForLocalization(image,(category,(x,y,w,h)))
        cv2.imwrite(outputPath + "/" + str(i) + "_" + str(j) + "_" + name,
                    newimage)

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
# class FolderLinearClassificationAugmentor:
#
#     def __init__(self,inputPath,outputPath):
#         IAugmentor.__init__(self)
#         self.inputPath = inputPath
#         # output path represents the folder where the images will be stored
#         self.outputPath = outputPath
#         self.generators = []
#
#     def addGenerator(self, generator):
#         self.generators.append(generator)
#
#     def readImagesAndAnnotations(self):
#         self.imagePaths = list(paths.list_images(self.inputPath))
#         self.labelPaths = list(paths.list_files(self.inputPath,validExts=(".xml")))
#         if (len(self.imagePaths) != len(self.labelPaths)):
#             raise Exception("The number of images is different to the number of annotations")
#
#     def applyAugmentation(self):
#         self.readImagesAndAnnotations()
#         Parallel(n_jobs=-1)(delayed(readAndGenerateImage)(self.outputPath,self.generators,x) for x in enumerate(self.imagePaths))
#
#
# # Example
# augmentor = FolderLinearClassificationAugmentor(
#     "/home/joheras/datasets/estomas/",
#     "/home/joheras/datasets/cats_and_dogs_small/data-augmented-parallel/"
# )
#
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
