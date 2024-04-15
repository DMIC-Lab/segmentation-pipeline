import torch
from monai.networks.nets import SwinUNETR
from segpipe.pipelineComponents import *
import numpy as np
from pkg_resources import resource_filename
from segpipe.orient import orient

class segmentationPipeline:
    def __init__(self,device,weightPathOverrides = [None,None,None]):
        self.LRModelPath = resource_filename(__name__,"weights/lr.pt")
        self.rightModelPath = resource_filename(__name__,"weights/right.pt")
        self.leftModelPath = resource_filename(__name__,"weights/left.pt")
        if weightPathOverrides[0] is not None:
            self.LRModelPath = weightPathOverrides[0]
        if weightPathOverrides[1] is not None:
            self.rightModelPath = weightPathOverrides[1]
        if weightPathOverrides[2] is not None:
            self.leftModelPath = weightPathOverrides[2]
        self.LRModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=3, feature_size=12)
        self.LRModel.load_state_dict(torch.load(self.LRModelPath,map_location=device))
        self.LRModel.eval()
        self.rightModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=4, feature_size=24)
        self.rightModel.load_state_dict(torch.load(self.rightModelPath,map_location=device))
        self.rightModel.eval()
        self.leftModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=3, feature_size=24)
        self.leftModel.load_state_dict(torch.load(self.leftModelPath,map_location=device))
        self.leftModel.eval()
        self.device = device
        self.LRModel.to(self.device)
        self.rightModel.to(self.device)
        self.leftModel.to(self.device)
    
    def segment(self,originalImage, getLR = 0,takeLargest=False, debug = False, returnImage= False, orientImage=False,postprocess=False):
        originalType = None
        if isinstance(originalImage, np.ndarray):
            if orientImage:
                originalImage = orient(originalImage)
            if returnImage:
                imageToReturn = originalImage
            originalImage = torch.from_numpy(originalImage).float()
            originalType = 'np'
        elif isinstance(originalImage, torch.Tensor):
            if orientImage:
                originalImage = orient(originalImage.cpu().numpy())
                originalImage = torch.tensor(originalImage)
            if returnImage:
                imageToReturn = originalImage
            originalImage = originalImage.float()
        else:
            raise TypeError("Input must be numpy array or torch tensor")
        
        if len(originalImage.shape) == 3:
            originalImage = originalImage.unsqueeze(0).unsqueeze(0)
        elif len(originalImage.shape) == 4:
            originalImage = originalImage.unsqueeze(0)
        elif len(originalImage.shape) == 5:
            pass
        else:
            raise ValueError("Input must be 3D, 4D, or 5D tensor")
        
        originalImage = originalImage.to(self.device)
        im = torch.sum(originalImage,dim=2).unsqueeze(2)
        im = im.repeat(1,1,originalImage.shape[2],1,1)
        im = torch.nn.functional.interpolate(im,size=(128,128,128),mode='nearest')
        eroded = pytorchBinaryErosion((im/torch.max(im)) > .5,selem_radius=3)
        dilated = pytorchBinaryDilation(eroded,selem_radius=3)
        dusted = torch.nn.functional.interpolate(dilated.to(torch.uint8),size=(originalImage.shape[2],originalImage.shape[3],originalImage.shape[4]),mode='nearest')
        originalImage = torch.where(dusted > 0, originalImage, torch.zeros_like(originalImage))

        lrImage = torch.nn.functional.interpolate(originalImage,size=(128,128,128),mode='nearest')
        
        #Segment LR - LR model outputs mask of what is left and right lung
        LRInput = torchPrep(lrImage) #HWD -> NCHWD
        LROutput = torchGetModelOutput(LRInput,self.LRModel)
        LROutput = torchDust(LROutput,threshold=5000,takeLargest=takeLargest)
        LROutput = torch.nn.functional.interpolate(LROutput, size=originalImage.shape[2:], mode='nearest')
        

        if getLR == 1:
            if originalType == 'np':
                if returnImage:
                    return LROutput.squeeze(0).squeeze(0).cpu().numpy(), imageToReturn
                return LROutput.squeeze(0).squeeze(0).cpu().numpy()
            if returnImage:
                return LROutput, imageToReturn
            return LROutput


        leftOutput = torch.where(LROutput==1,1,0)
        rightOutput = torch.where(LROutput==2,1,0)
        if torch.sum(leftOutput) == 0:
            leftOutput = None
        if torch.sum(rightOutput) == 0:
            rightOutput = None
        if leftOutput is None and rightOutput is None:
            print("No lungs detected")
            return None
        if leftOutput is not None:
            leftBounds = torchbbox2_3D(leftOutput,margin=1)
            leftCropped = torchCrop(originalImage,leftBounds)
        else:
            leftCropped = None
        if rightOutput is not None:
            rightBounds = torchbbox2_3D(rightOutput,margin=1)
            rightCropped = torchCrop(originalImage,rightBounds)
        else:
            rightCropped = None

        # debug = torch.zeros(originalImage.shape).cuda()
        # debug[:,:,leftBounds[0]:leftBounds[1],leftBounds[2]:leftBounds[3],leftBounds[4]:leftBounds[5]] = 255
        # debug[:,:,rightBounds[0]:rightBounds[1],rightBounds[2]:rightBounds[3],rightBounds[4]:rightBounds[5]] = 255
        # debug = debug.to(torch.uint8).squeeze(0).squeeze(0)
        # debug = debug.cpu().numpy()
        # return debug



        if leftCropped is not None:
            # Get and post-process left lobe model output
            leftInput = torchPrep(leftCropped)
            leftLobeOutput = torchGetModelOutput(leftInput,self.leftModel)
            leftLobeOutput = torch.nn.functional.interpolate(leftLobeOutput, size=leftCropped.shape[2:], mode='nearest')
            temp = torch.zeros_like(leftLobeOutput)
            temp[:,:,:-1,:-1,:-1] = leftLobeOutput[:,:,1:,1:,1:]
            leftLobeOutput = temp
        if rightCropped is not None:
            # Get and post-process right lobe model output
            rightInput = torchPrep(rightCropped)
            rightLobeOutput = torchGetModelOutput(rightInput,self.rightModel)       
            rightLobeOutput = torch.nn.functional.interpolate(rightLobeOutput, size=rightCropped.shape[2:], mode='nearest')

            #adjust right lobe output (0,1,2,3) to (0,3,4,5)
            rightLobeOutput = rightLobeOutput + 2
            rightLobeOutput = torch.where(rightLobeOutput==2,0,rightLobeOutput)
            temp = torch.zeros_like(rightLobeOutput)
            temp[:,:,:-1,:-1,:-1] = rightLobeOutput[:,:,1:,1:,1:]
            rightLobeOutput = temp
        

        #Assemble final mask
        finalMask = torch.zeros(originalImage.shape).cuda()
        leftFullSize = torch.zeros(originalImage.shape).cuda()
        rightFullSize = torch.zeros(originalImage.shape).cuda()
        if leftCropped is not None:
            leftFullSize[:,:,leftBounds[0]:leftBounds[1],leftBounds[2]:leftBounds[3],leftBounds[4]:leftBounds[5]] = leftLobeOutput
            finalMask = torch.where(leftFullSize > 0, leftFullSize, finalMask)
        if rightCropped is not None:
            rightFullSize[:,:,rightBounds[0]:rightBounds[1],rightBounds[2]:rightBounds[3],rightBounds[4]:rightBounds[5]] = rightLobeOutput
            finalMask = torch.where(rightFullSize > 0, rightFullSize, finalMask)

        unevenShape = [False,False,False]
        if finalMask.shape[2] % 2 != 0:
            unevenShape[0] = True
        if finalMask.shape[3] % 2 != 0:
            unevenShape[1] = True
        if finalMask.shape[4] % 2 != 0:
            unevenShape[2] = True
        shape = list(finalMask.shape)
        for i in range(3):
            if unevenShape[i]:
                shape[i+2] += 1
        finalMask = torch.nn.functional.interpolate(finalMask, size=shape[2:], mode='nearest-exact')
        if postprocess:
            finalMask = torchErrors(finalMask)
        finalMask = torchDust(finalMask)
        #finalMask = torchSmoothing(finalMask)

        finalMask = torch.nn.functional.interpolate(finalMask, size=originalImage.shape[2:], mode='nearest-exact')
        
        finalMask = finalMask.to(torch.uint8)
        
        if originalType == 'np':
            if returnImage:
                return finalMask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8), imageToReturn
            finalMask = finalMask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        if getLR == 2:
            finalMask = np.where((finalMask == 1)|(finalMask == 2),1,finalMask)
            finalMask = np.where((finalMask == 3)|(finalMask == 4)|(finalMask == 5),2,finalMask)

        if returnImage:
            return finalMask, imageToReturn
        return finalMask