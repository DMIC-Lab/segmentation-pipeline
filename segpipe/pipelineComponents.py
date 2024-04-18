import torch
from skimage import morphology
from cc_torch import connected_components_labeling


def pytorchGetLargest(tensor, num = None, threshold=3000, ignoreBackground=True,override_comp_id=None):
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    if len(tensor.shape) == 5:
        tensor = tensor.squeeze(0).squeeze(0)
    if tensor.dtype != torch.uint8:
        tensor = tensor.to(torch.uint8)
    test = (tensor > 0).to(torch.uint8)
    connected_components = connected_components_labeling(test)
    if override_comp_id is None:
        volume_count = {}
        for component in connected_components.unique():
            if component == 0 and ignoreBackground:  # background
                continue
            count = torch.sum(connected_components == component).item()
            if count < threshold:
                continue
            volume_count[component.item()] = count
        if num is None:
            num = len(volume_count)
        # Sort by volume
        sorted_components = sorted(volume_count.items(), key=lambda x: x[1], reverse=True)

        # Take two largest components
        largest_components = [comp_id for comp_id, _ in sorted_components[:num]]
    else:
        largest_components = override_comp_id

    # Create a new 3D array with only the two largest components
    new_array_3d = torch.zeros_like(tensor)
    for comp_id in largest_components:
        new_array_3d[connected_components == comp_id] = tensor[connected_components == comp_id]   # or comp_id to retain original labels

    return new_array_3d.unsqueeze(0).unsqueeze(0)
    
def torchGetModelOutput(input,model):
    softmax = torch.nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(input)
        output = softmax(output)
        output = torch.argmax(output,dim=1).unsqueeze(0)
    return output.to(torch.uint8)

def pytorchBinaryErosion(tensor, selem_radius=3):
    ball = morphology.ball(selem_radius)
    struct_elem = torch.tensor(ball, dtype=torch.float32)
    struct_elem = struct_elem.view(1, 1, *struct_elem.size()).cuda()

    # Perform 3D convolution with structuring element
    conv_result = torch.nn.functional.conv3d(tensor.float(), struct_elem, padding=selem_radius)

    # Binary erosion is equivalent to finding where the convolution result
    # is equal to the sum of the structuring element
    erosion_result = conv_result == struct_elem.sum().item()
    return erosion_result

def torchMorphology(smallTensor):
    thresholdedTensor = (smallTensor > 128)
    flippedTensor = torch.logical_not(thresholdedTensor).to(torch.uint8)
    ccTensor = pytorchGetLargest(flippedTensor,ignoreBackground=False,override_comp_id=[flippedTensor[:,:,0,0,0].item()])
    finalTensor = pytorchBinaryErosion(torch.logical_not(ccTensor))
    return finalTensor

def torchbbox2_3D(img,margin=5):
    img = img.bool()
    if len(img.shape) > 3:
        for i in range(len(img.shape)-3):
            img = img.squeeze(0)
    r = img.any(dim=2).any(dim=1)
    c = img.any(dim=2).any(dim=0)
    z = img.any(dim=1).any(dim=0)
    xmin, xmax = torch.where(r)[0][[0, -1]]
    ymin, ymax = torch.where(c)[0][[0, -1]]
    zmin, zmax = torch.where(z)[0][[0, -1]]
    
    xmin = max(0,xmin-margin)
    xmax = min(img.shape[0],xmax+margin)
    ymin = max(0,ymin-margin)
    ymax = min(img.shape[1],ymax+margin)
    zmin = max(0,zmin-margin)
    zmax = min(img.shape[2],zmax+margin)
    return int(xmin), int(xmax), int(ymin), int(ymax), int(zmin), int(zmax)

def torchPrep(image):
    image = torch.nn.functional.interpolate(image,size=(128,128,128),mode='nearest')
    return image

def pytorchBinaryDilation(tensor, selem_radius=3):
    ball = morphology.ball(selem_radius)
    struct_elem = torch.tensor(ball, dtype=torch.float32)
    struct_elem = struct_elem.view(1, 1, *struct_elem.size()).cuda()

    # Perform 3D convolution with structuring element
    conv_result = torch.nn.functional.conv3d(tensor.float(), struct_elem, padding=selem_radius)

    # Binary dilation is equivalent to finding where the convolution result
    # is greater than 0
    dilation_result = conv_result > 0

    return dilation_result

def torchCrop(originalImage, bounds):
    return originalImage[:,:,bounds[0]:bounds[1],bounds[2]:bounds[3],bounds[4]:bounds[5]]

def torchRescaleBounds(bounds,originalImageShape,imageShape):
    bounds = list(bounds)
    xScale = originalImageShape[2]/imageShape[2]
    yScale = originalImageShape[3]/imageShape[3]
    zScale = originalImageShape[4]/imageShape[4]
    bounds[0] = int(bounds[0]*xScale)
    bounds[1] = int(bounds[1]*xScale)
    bounds[2] = int(bounds[2]*yScale)
    bounds[3] = int(bounds[3]*yScale)
    bounds[4] = int(bounds[4]*zScale)
    bounds[5] = int(bounds[5]*zScale)
    return bounds


def fitInBounds(rescaledBounds, intermediateImageBounds):
    finalBounds = list(rescaledBounds)
    
    # The intermediateImageBounds should have format [x_min, x_max, y_min, y_max, z_min, z_max]
    
    # adjust the x coordinates
    finalBounds[0] = rescaledBounds[0] + intermediateImageBounds[0]
    finalBounds[1] = rescaledBounds[1] + intermediateImageBounds[0]
    
    # adjust the y coordinates
    finalBounds[2] = rescaledBounds[2] + intermediateImageBounds[2]
    finalBounds[3] = rescaledBounds[3] + intermediateImageBounds[2]
    
    # adjust the z coordinates
    finalBounds[4] = rescaledBounds[4] + intermediateImageBounds[4]
    finalBounds[5] = rescaledBounds[5] + intermediateImageBounds[4]
    
    return finalBounds

def multiLabelCC(arr):
    ccs = torch.zeros([len(arr.unique()),*arr.shape],dtype=torch.uint8,device=arr.device)
    for i,component in enumerate(arr.unique()):
        ccs[i] = connected_components_labeling((arr==component).to(torch.uint8))
    cc = torch.zeros_like(arr,device=arr.device)
    vals = [1,3,5,7,9,11]
    for i in range(len(arr.unique())):
        cc += ccs[i]*(vals[i])


    return cc

def torchDust(arr,threshold=3000,takeLargest=False):
    originalShapeLen = len(arr.shape)
    if originalShapeLen > 3:
        for i in range(originalShapeLen-3):
            arr = arr.squeeze(0)
    dusted = torch.zeros_like(arr,device=arr.device)
    cc = multiLabelCC(arr)
    if takeLargest:
        largestDict = {val.item():0 for val in arr.unique()}
        componentDict = {val.item():None for val in arr.unique()}

    totalCount = 0
    for component in cc.unique():
        
        count = torch.sum(cc == component).item()
        if count < threshold:
            totalCount += count
            continue
        if takeLargest:
            if largestDict[arr[cc == component][0].item()] < count:
                largestDict[arr[cc == component][0].item()] = count
                componentDict[arr[cc == component][0].item()] = component
                #dusted[cc == component] = arr[cc == component]
            else:
                totalCount += count
        else:
            dusted[cc == component] = arr[cc == component]
    if takeLargest:
        for key, val in componentDict.items():
            if val is not None:
                dusted[cc == val] = key
            else:
                print(componentDict)
                print('failed')
                return None
    for i in range(originalShapeLen-3):
        dusted = dusted.unsqueeze(0)
    return dusted
    

def torchErrors(arr,threshold=3000,takeLargest=False):
    originalShape = arr.shape[2:]
    newShape = [val-1 if val > 1 and val%2 != 0 else val for val in originalShape]
    arr = torch.nn.functional.interpolate(arr,size=newShape,mode='nearest')
    arr = arr.squeeze(0).squeeze(0).to(torch.uint8)
    arrDusted = torchDust(arr,threshold=threshold,takeLargest=takeLargest)
    if arrDusted is None:
        return None
    errors = torch.logical_xor(arrDusted,arr)
    isolatedErrors = torch.where(errors,arr,0)
    isolatedErrorCC = multiLabelCC(isolatedErrors)
    for id in torch.unique(isolatedErrorCC):
        
        region = torch.where(isolatedErrorCC==id,arr,0).to(torch.bool)
        if torch.sum(region).item() > 100000:
            continue
        dilatedRegion = pytorchBinaryDilation(region.unsqueeze(0).unsqueeze(0),selem_radius=3).squeeze(0).squeeze(0)
        boundary = torch.logical_and(dilatedRegion, torch.logical_not(region))
        boundary = torch.where(boundary,arr,0)
        regionVals = torch.where(region,arr,0)
        boundaryUnique = set([val.item() for val in boundary.unique()])
        regionUnique = set([val.item() for val in regionVals.unique()])
        uniqueLabels = boundaryUnique.difference(regionUnique)
        
        if len(uniqueLabels) == 1:
            arr = torch.where(region,list(uniqueLabels)[0],arr)
            #viewer.add_image(arr.cpu().squeeze(0).squeeze(0).numpy(),name='corrected',colormap="gist_earth",contrast_limits=(0,5))
        elif len(uniqueLabels) == 0:
            arr = torch.where(region,0,arr)
        else:
            sizes = {val:torch.sum(boundary==val).item() for val in uniqueLabels if val != 0}
            largest = max(sizes, key=sizes.get)
            arr = torch.where(region,largest,arr)
    arr = torch.nn.functional.interpolate(arr.unsqueeze(0).unsqueeze(0).float(),size=originalShape,mode='nearest')
    return arr

# def edge_rounding(finalMask):
    
#     # create newMask w/ same shape of finalMask filled with zeroes
#     newMask = np.zeros_like(finalMask, np.uint8)

#     # taking a 2D matrix of size (1, 1) as the kernel
#     kernel = np.ones((1, 1), np.uint8)
    
#     # extract each lobe
#     for i in range(6):
#         # separate background
#         if i == 0:
#             continue

#         # get binary mask of current lobe
#         binaryMask = (finalMask == i).astype(np.uint8)

#         # initialize an empty 3D array for dilated and eroded masks
#         erodedMask = np.zeros_like(binaryMask)

#         # iterate over each 2D slice along the third dimension
#         for j in range(binaryMask.shape[2]):
#             # dilate and erode
#             erodedMask = cv2.dilate(binaryMask, kernel, iterations=1)
#             erodedMask = cv2.dilate(binaryMask, kernel, iterations=1)
#             erodedMask = cv2.dilate(binaryMask, kernel, iterations=1)
#             erodedMask = cv2.erode(binaryMask, kernel, iterations=1)
#             erodedMask = cv2.erode(binaryMask, kernel, iterations=1)
#             erodedMask = cv2.erode(binaryMask, kernel, iterations=1)
#             print(j)

#         #set new mask to value of current lobe
#         newMask += i * erodedMask

#     return newMask

def torchSmoothing(finalMask):

    newMask = torch.zeros_like(finalMask, dtype=torch.uint8)

    for i in range(6):
        if i == 0:
            continue

        binaryMask = (finalMask == i).to(torch.uint8)

        erodedMask = torch.zeros_like(binaryMask)

        #for j in range(binaryMask.shape[2]):
        erodedMask = pytorchBinaryDilation(binaryMask, selem_radius=1)
        erodedMask = pytorchBinaryDilation(erodedMask, selem_radius=1)
        erodedMask = pytorchBinaryDilation(erodedMask, selem_radius=1)
        erodedMask = pytorchBinaryErosion(erodedMask, selem_radius=1)
        erodedMask = pytorchBinaryErosion(erodedMask, selem_radius=1)
        erodedMask = pytorchBinaryErosion(erodedMask, selem_radius=1)
        newMask = torch.where(erodedMask > 0, i, newMask)
    return newMask