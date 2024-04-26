from segpipe import segmentationPipeline
import numpy as np

pipeline = segmentationPipeline("cuda:0")
testImage = np.load('test.npy')
#testImage = loadmat('/home/aaronluong/Downloads/B10.mat')['T00']
output = pipeline.segment(testImage)