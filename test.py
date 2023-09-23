from pipeline import segmentationPipeline
import nibabel as nib
import numpy as np

pipeline = segmentationPipeline("cuda:0")
testImage = nib.load("./10032T_EXP_image.nii.gz").get_fdata()
testImage = 1+ testImage / 1000
testImage = np.clip(testImage,0,1) * 255
testImage = testImage.astype(np.uint8)
pipeline.segment(testImage)