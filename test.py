from pipeline import segmentationPipeline
import nibabel as nib
import numpy as np
import napari

pipeline = segmentationPipeline("cuda:0")
testImage = np.load('test.npy')
output = pipeline.segment(originalImage=testImage).squeeze(0).squeeze(0).cpu().numpy()
napari.view_image(output,colormap='gist_earth',contrast_limits=(0,5))
napari.run()