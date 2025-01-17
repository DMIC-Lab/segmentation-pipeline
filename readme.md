# TriSwinUNETR-lobeseg

This Python module provides a simple-to-use interface for lung segmentation tasks using deep learning. Specifically, this module includes a class `segmentationPipeline` that allows for the segmentation of lung lobes and left-right lung separation.

## Requirements

- Python
- PyTorch
- Numpy
- Monai
- Connected Components Labeling for Pytorch
- Connected Components 3D (Numpy-based fallback)
- OpenCV

## Installing Requirements
Make a new environment and then install the following packages

Install torch (https://pytorch.org/get-started/locally/)

pip install monai[einops]

OPTIONALLY
git clone https://github.com/zsef123/Connected_components_PyTorch
cd Connected_components_Pytorch


pip install .


## Quickstart Guide

Here's how to get started:

### Importing the module

First, import the `segmentationPipeline` from the pipeline:

```python
from segpipe import segmentationPipeline
```

### Initializing the pipeline

You can initialize the pipeline by passing a PyTorch device, and optionally, the paths to the model weights you want to use:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optionally specify custom model weight paths
weight_paths = ['LR.pth', 'right.pth', 'left.pth']

pipeline = segmentationPipeline(device, weight_paths)
```

### Running the segmentation

You can segment an image (as a PyTorch tensor or a Numpy array) of shape (N,C,H,W,D), N and C optional,  by calling the `segment` method:
The model assumes the image is already oriented Inferior-Superior, Anterior-Posterior, Right-Left, scaled to 0 to 1 where 0 is air and 1 is water density.
```python
input_image = torch.rand((1, 1,512, 512, 512))  # Replace with your own image tensor

# Lobe segmentation
result_lobe = pipeline.segment(input_image, getLR=0)

# Rough left-right lung segmentation
result_LR = pipeline.segment(input_image, getLR=1)

# Lobe segmentation cast down to left and right lung
result_cast_LR = pipeline.segment(input_image, getLR=2)

#Use numpy based dusting
result = pipeline.segment(input_image,numpyDust=True)
```


## API

### `segmentationPipeline`

#### Methods

- `__init__(device, weight_paths=None)`: Initializes the pipeline.
  - `device`: A PyTorch device (e.g., `torch.device('cuda')`).
  - `weight_paths`: Optional. A list of paths to custom weights in the order [LRPath, rightPath, leftPath].

- `segment(image, getLR)`: Segments the given image.
  - `image`: Input image as a PyTorch tensor or a Numpy array.
  - `getLR`: Determines the type of segmentation.
    - `0`: Lobe Segmentation
    - `1`: Rough Left/Right Segmentation
    - `2`: Left/Right Segmentation Derived from Lobes
  - `numpyDust`: Use numpy based dusting (slightly slower but more reliable)
