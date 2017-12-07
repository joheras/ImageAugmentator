from averageBlurringAugmentationTechnique import averageBlurringAugmentationTechnique
from bilateralBlurringAugmentationTechnique import bilateralBlurringAugmentationTechnique
from blurringAugmentationTechnique import blurringAugmentationTechnique
from cropAugmentationTechnique import cropAugmentationTechnique
from dropoutAugmentationTechnique import dropoutAugmentationTechnique
from elasticTransformAugmentationTechnique import elasticTransformAugmentationTechnique
from equalizeHistogramAugmentationTechnique import equalizeHistogramAugmentationTechnique
from flipAugmentationTechnique import flipAugmentationTechnique
from gammaCorrectionAugmentationTechnique import gammaCorrectionAugmentationTechnique
from gaussianBlurringAugmentationTechnique import gausianBlurringAugmentationTechnique
from gaussianNoiseAugmentationTechnique import gaussianNoiseAugmentationTechnique
from medianBlurringAugmentationTechnique import medianBlurringAugmentationTechnique
from noneAugmentationTechnique import noneAugmentationTechnique
from raiseBlueAugmentationTechnique import raiseBlueAugmentationTechnique
from raiseGreenAugmentationTechnique import raiseGreenAugmentationTechnique
from raiseHueAugmentationTechnique import raiseHueAugmentationTechnique
from raiseRedAugmentationTechnique import raiseRedAugmentationTechnique
from raiseSaturationAugmentationTechnique import raiseSaturationAugmentationTechnique
from raiseValueAugmentationTechnique import raiseValueAugmentationTechnique
from resizeAugmentationTechnique import resizeAugmentationTechnique
from rotateAugmentationTechnique import rotateAugmentationTechnique
from saltAndPepperNoiseAugmentationTechnique import saltAndPepperNoiseAugmentationTechnique
from sharpenAugmentationTechnique import sharpenAugmentationTechnique
from shearingAugmentationTechnique import shearingAugmentationTechnique
from translationAugmentationTechnique import translationAugmentationTechnique


Techniques = {
    "average_blurring" : averageBlurringAugmentationTechnique,
    "bilateral_blurring" : bilateralBlurringAugmentationTechnique,
    "blurring" : blurringAugmentationTechnique,
    "crop" : cropAugmentationTechnique,
    "dropout" : dropoutAugmentationTechnique,
    "elastic" : elasticTransformAugmentationTechnique,
    "equalize_histogram" : equalizeHistogramAugmentationTechnique,
    "flip" : flipAugmentationTechnique,
    "gamma" : gammaCorrectionAugmentationTechnique,
    "gaussian_blur": gausianBlurringAugmentationTechnique,
    "gaussian_noise": gaussianNoiseAugmentationTechnique,
    "median_blur": medianBlurringAugmentationTechnique,
    "none":  noneAugmentationTechnique,
    "raise_blue":  raiseBlueAugmentationTechnique,
    "raise_green":raiseGreenAugmentationTechnique,
    "raise_hue": raiseHueAugmentationTechnique,
    "raise_red": raiseRedAugmentationTechnique,
    "raise_saturation": raiseSaturationAugmentationTechnique,
    "raise_value":raiseValueAugmentationTechnique,
    "resize": resizeAugmentationTechnique,
    "rotate": rotateAugmentationTechnique,
    "salt_and_pepper":saltAndPepperNoiseAugmentationTechnique,
    "sharpen":sharpenAugmentationTechnique,
    "shearing": shearingAugmentationTechnique,
    "translation": translationAugmentationTechnique
}

def createTechnique(technique,parameters):
    Technique = Techniques[technique]
    if Technique is None:
        raise ValueError("That technique is not available")
    return Technique(parameters)