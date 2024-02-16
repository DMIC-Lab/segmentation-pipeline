from pipeline import segmentationPipeline
import nibabel as nib
import numpy as np
import napari
from scipy.io import loadmat

pipeline = segmentationPipeline("cuda:0")
testImage = np.load('test.npy')
#testImage = loadmat('/home/aaronluong/Downloads/B10.mat')['T00']
output = pipeline.segment(originalImage=testImage,takeLargest=False)
viewer = napari.view_image(testImage)
viewer.add_image(output,blending='additive',colormap='gist_earth',contrast_limits=(0,5))
napari.run()