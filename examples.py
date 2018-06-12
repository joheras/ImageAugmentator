from techniqueFactory import createTechnique
import cv2











"average_blurring" : averageBlurringAugmentationTechnique,
    "bilateral_blurring" : bilateralBlurringAugmentationTechnique,
    "blurring" : blurringAugmentationTechnique,
    "change_to_hsv" : changeToHSVAugmentationTechnique,
    "change_to_lab" : changeToLABAugmentationTechnique,
    "crop" : cropAugmentationTechnique,
    "dropout" : dropoutAugmentationTechnique,
    "elastic" : elasticTransformAugmentationTechnique,
    "equalize_histogram" : equalizeHistogramAugmentationTechnique,
    "flip" : flipAugmentationTechnique,
    "gamma" : gammaCorrectionAugmentationTechnique,
    "gaussian_blur": gausianBlurringAugmentationTechnique,
    "gaussian_noise": gaussianNoiseAugmentationTechnique,
    "invert": invertAugmentationTechnique,
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
    "shift_channel":shiftChannelAugmentationTechnique,
    "shearing": shearingAugmentationTechnique,
    "translation": translationAugmentationTechnique

