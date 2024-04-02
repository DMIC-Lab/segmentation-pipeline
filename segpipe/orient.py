
import numpy as np
import cc3d
from scipy.ndimage import zoom
from skimage.morphology import binary_dilation, binary_erosion
import cv2
# def identifyAxial(image):
#     areas = []
#     for i in range(3):
#         maybeSpine = np.sum(image,axis=i)
#         body = cropToBody(maybeSpine)
#         area = np.sum(body > np.max(body)*.9)
#         areas.append(area)
#     index = np.argmin(areas)
#     return index

def find_starting_value(img):
    # Flatten the image and find unique values
    unique_values = np.unique(img.flatten())
    # Find the starting point of the more continuous distribution
    min_gap = np.inf
    starting_value = None
    for i in range(len(unique_values) - 1):
        gap = unique_values[i + 1] - unique_values[i]
        if gap < min_gap:
            min_gap = gap
            starting_value = unique_values[i]
        else:
            return starting_value
    return starting_value

def prepForOrientation(img):
    if np.max(img) > 1000:
        starting_value = find_starting_value(img)
        img = img.astype(float)
        img -= starting_value
    else:
        print('Image assumed to be already preprocessed, please correct manually if not')
    return img


def identifyAxialBone(image):
    #to fix - threshold should be a function of the image intensity distribution
    bone = np.where(image > 1700,image,0)

    boneProj0 = np.sum(bone,axis=0)
    boneProj1 = np.sum(bone,axis=1)
    boneProj2 = np.sum(bone,axis=2)

    axialEval0 = not (((np.sum(boneProj0,axis=0) > 0).all()) or ((np.sum(boneProj0,axis=1) > 0).all()))
    axialEval1 = not (((np.sum(boneProj1,axis=0) > 0).all()) or ((np.sum(boneProj1,axis=1) > 0).all()))
    axialEval2 = not (((np.sum(boneProj2,axis=0) > 0).all()) or ((np.sum(boneProj2,axis=1) > 0).all()))
    
    results = [axialEval0,axialEval1,axialEval2]
    if sum(results) != 1:
        return -1
    else:
        return results.index(True)


def boundingBox(arr):
    x = np.any(arr, axis=(1, 2))
    y = np.any(arr, axis=(0, 2))
    z = np.any(arr, axis=(0, 1))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return xmin, xmax, ymin, ymax, zmin, zmax

def twoDBoundingBox(arr):
    x = np.any(arr, axis=(1))
    y = np.any(arr, axis=(0))

    xmin, xmax = np.where(x)[0][[0, -1]]
    ymin, ymax = np.where(y)[0][[0, -1]]
    return xmin, xmax, ymin, ymax

def cropToBody(image):
    #to fix - threshold should be a function of the image size
    probablyBody = cc3d.dust(image > np.max(image) / 3, threshold=5000, connectivity=8)
    bounds = twoDBoundingBox(probablyBody)
    return image[bounds[0]:bounds[1], bounds[2]:bounds[3]]
def getRelativeDistanceFromMidpoint(spineArea):
    spineCentroid = np.mean(np.where(spineArea),axis=1)
    xmid = spineArea.shape[0] // 2
    ymid = spineArea.shape[1] // 2
    return (spineCentroid[0] - xmid) / xmid, (spineCentroid[1] - ymid) / ymid
def getAxialK(spineArea):
    xr,yr = getRelativeDistanceFromMidpoint(spineArea)
    #print(xr,yr)
    if abs(xr) > abs(yr):
        if xr > 0:
            #print('aligned')
            return 0
        else:
            #print('top')
            return 2
    else:
        if yr > 0:
            #print('right')
            return 3
        else:
            #print('left')
            return 1
        
def rotateImagewithAxialK(image,axialK):
        return np.rot90(image,k=axialK,axes=(1,2))
    
    
def transposeImage(image,axialIndex):
    if axialIndex == 0:
        return image
    elif axialIndex == 1:
        return np.transpose(image,(1,2,0))
    else:
        return np.transpose(image,(2,0,1))
    

def orient(image):
    image = prepForOrientation(image)
    axialIndex = identifyAxialBone(image)
    if axialIndex == -1:
        print('Could not identify axial index')
        return None
    transposedImage = transposeImage(image,axialIndex)
    spine = np.sum(transposedImage,axis=0)
    spineImage = cropToBody(spine)
    #to fix - threshold should be a function of the image size
    spineArea = cc3d.dust(spineImage > np.max(spineImage) * .8, threshold = 1000, connectivity=8)
    axialK = getAxialK(spineArea)
    rotatedImage = rotateImagewithAxialK(transposedImage,axialK)
    trachea = rotatedImage < 300
    originalShape = trachea.shape
    trachea = zoom(trachea,(128/trachea.shape[0],128/trachea.shape[1],128/trachea.shape[2]),order=1)
    for i in range(4):
        trachea = binary_dilation(trachea)    
    for i in range(4):
        trachea = binary_erosion(trachea)
    dusted = cc3d.dust(trachea,threshold=1000,connectivity=26)
    cc = cc3d.connected_components(dusted)
    middle = [val // 2 for val in trachea.shape]
    #get connected component closest to middle
    minDist = 100000000
    minIndex = 0
    for i,loc in cc3d.each(cc,binary=False,in_place=True):
        loc = np.array(np.where(loc)).T
        np.random.shuffle(loc)
        loc = loc[:1000]
        dist = np.apply_along_axis(lambda x: np.linalg.norm(x - middle),1,loc).mean()
        if dist < minDist:
            minDist = dist
            minIndex = i
    localized = cc == minIndex
    localized = zoom(localized,(originalShape[0]/localized.shape[0],originalShape[1]/localized.shape[1],originalShape[2]/localized.shape[2]),order=1)
    localized = np.sum(localized,axis=1)
    smoothLungs = cv2.GaussianBlur(localized.astype(np.uint8),(5,5),0) * 255
    lungEdges = cv2.Canny(smoothLungs.astype(np.uint8),100,200)
    contours, _ = cv2.findContours(lungEdges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hulls = [cv2.convexHull(contour) for contour in contours]
    outputImage = np.zeros([*localized.shape,3],dtype=np.uint8)  # Convert to 3-channel image for visualization
    for hull in hulls:
        cv2.drawContours(outputImage, [hull], 0, (255, 255, 255), -1)  # Draw filled contour
    diff = outputImage[:,:,0] - smoothLungs
    diffLine = np.sum(diff,axis=1)
    diffLineTrimmed = diffLine[diffLine > 0]
    middleLength = len(diffLineTrimmed) // 2
    upsideDown = np.argmax(diffLineTrimmed) > middleLength
    if upsideDown:
        flippedImage = np.flip(rotatedImage,0)
    else:
        flippedImage = rotatedImage
    im = np.sum(flippedImage,axis=1) 
    im = im / np.max(im) * 255
    line = np.sum(im,axis=0)
    removed = line > .9*np.max(line)
    removedLeftBound = np.argmax(removed)
    removedRightBound = len(removed) - np.argmax(removed[::-1])
    line[line > .9*np.max(line)] = 0
    # plt.plot(line)
    # plt.show()
    left = line[removedRightBound:]
    right = line[:removedLeftBound]
    left = left[len(left)//2:]
    right = right[:len(right)//2]
    leftSum = np.sum(left)
    rightSum = np.sum(right)
    flip = leftSum > rightSum
    if flip:
        flippedImage = np.flip(flippedImage,2)
    finalImage = np.clip(flippedImage / 1000,0,1)
    return finalImage



                