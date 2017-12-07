from augmentors.augmentorFactory import createAugmentor
from augmentors.generator import Generator
from techniques.techniqueFactory import createTechnique
import argparse
from utils.conf import Conf
import cv2
from mosaic import generateMosaic

ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="path to configuration file")
ap.add_argument("-i", "--image", required=True, help="path to the image to generate the sample")
args = vars(ap.parse_args())


conf = Conf(args["conf"])
image = cv2.imread(args["image"])

# First, we read the parameters
problem = conf["problem"]
annotationMode = conf["annotation_mode"]
outputMode = conf["output_mode"]
generationMode = conf["generation_mode"]
inputPath = conf["input_path"]
parameters = conf["parameters"]
augmentationTechniques = conf["augmentation_techniques"]

# Second, we create the augmentor
augmentor = createAugmentor(problem,annotationMode,outputMode,generationMode,inputPath,
                            parameters)

# Third, we load the techniques and add them to the augmentor
techniques = [createTechnique(technique,parameters) for (technique,parameters) in
              augmentationTechniques]

# We apply the augmentation
images = []
for technique in techniques:
    images.append(Generator(technique).applyForClassification(image))

cv2.imshow("Mosaic",generateMosaic(images))
cv2.waitKey(0)

