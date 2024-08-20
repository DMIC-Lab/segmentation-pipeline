from segpipe import segmentationPipeline
import numpy as np

pipeline = segmentationPipeline("cpu")
testImage = np.random.rand(64,64,32)#np.load('test.npy')

#testImage = loadmat('/home/aaronluong/Downloads/B10.mat')['T00']
output = pipeline.segment(testImage)

print(output.shape, 'success')