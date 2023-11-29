import torch
from monai.networks.nets import SwinUNETR
from utils import *
import numpy as np

class segmentationPipeline:
    def __init__(self,device,weightPathOverrides = [None,None,None,None]):
        self.coarseModelPath = "./weights/coarse.pth"
        self.LRModelPath = "./weights/lr.pth"
        self.rightModelPath = "./weights/right.pth"
        self.leftModelPath = "./weights/left.pth"
        if weightPathOverrides[0] is not None:
            self.coarseModelPath = weightPathOverrides[0]
        if weightPathOverrides[1] is not None:
            self.LRModelPath = weightPathOverrides[1]
        if weightPathOverrides[2] is not None:
            self.rightModelPath = weightPathOverrides[2]
        if weightPathOverrides[3] is not None:
            self.leftModelPath = weightPathOverrides[3]
        self.coarseModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=2, feature_size=12)
        self.coarseModel.load_state_dict(torch.load(self.coarseModelPath))
        self.coarseModel.eval()
        self.LRModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=3, feature_size=12)
        self.LRModel.load_state_dict(torch.load(self.LRModelPath))
        self.LRModel.eval()
        self.rightModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=4, feature_size=12)
        self.rightModel.load_state_dict(torch.load(self.rightModelPath))
        self.rightModel.eval()
        self.leftModel = SwinUNETR(img_size=(128,128,128), in_channels=1, out_channels=3, feature_size=12)
        self.leftModel.load_state_dict(torch.load(self.leftModelPath))
        self.leftModel.eval()
        self.device = device
        self.coarseModel.to(self.device)
        self.LRModel.to(self.device)
        self.rightModel.to(self.device)
        self.leftModel.to(self.device)
    
    def segment(self,originalImage, getLR = 0):
        if isinstance(originalImage, np.ndarray):
            originalImage = torch.from_numpy(originalImage).float()
        elif isinstance(originalImage, torch.Tensor):
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
        originalImage = originalImage / (torch.max(originalImage) / 255)

        coarseImage = torch.nn.functional.interpolate(originalImage,size=(128,128,128),mode='nearest') / 255
        #Segment lung - coarse model outputs binary mask of what is lung and not
        coarseOutput = torchGetModelOutput(coarseImage,self.coarseModel)
        
        # bodyMask = torchMorphology(coarseImage)
        # coarseOutput = torch.where(bodyMask > 0,coarseOutput,0)
        coarseOutput = pytorchGetLargest(coarseOutput,num=2) #HWD
        
        coarseBounds128 = torchbbox2_3D(coarseOutput)
        coarseBounds = torchRescaleBounds(coarseBounds128,originalImage.shape,coarseOutput.shape)
        coarseCropped = torchCrop(originalImage,coarseBounds)
        
        #Segment LR - LR model outputs mask of what is left and right lung
        LRInput = torchPrep(coarseCropped) #HWD -> NCHWD
        LROutput = torchGetModelOutput(LRInput,self.LRModel)
        print('lr')
        LROutput = pytorchGetLargest(LROutput,num=2)
        LRFullSizeMask = torch.zeros(originalImage.shape).cuda()
        LRCoarseSize = torch.nn.functional.interpolate(LROutput,size=coarseCropped.shape[2:],mode='nearest-exact')
        LRFullSizeMask[:,:,coarseBounds[0]:coarseBounds[1],coarseBounds[2]:coarseBounds[3],coarseBounds[4]:coarseBounds[5]] = LRCoarseSize
        if getLR == 1:
            return LRFullSizeMask
        
        leftOutput = torch.where(LROutput==1,1,0)
        rightOutput = torch.where(LROutput==2,1,0)
        leftOutput = pytorchGetLargest(leftOutput,num=1)
        rightOutput = pytorchGetLargest(rightOutput,num=1)
        leftBounds128 = torchbbox2_3D(leftOutput,margin=1)
        rightBounds128 = torchbbox2_3D(rightOutput,margin=1)
        leftCoarseBounds = torchRescaleBounds(leftBounds128,coarseCropped.shape,leftOutput.shape)
        rightCoarseBounds = torchRescaleBounds(rightBounds128,coarseCropped.shape,rightOutput.shape)
        leftBounds = fitInBounds(leftCoarseBounds,coarseBounds)
        rightBounds = fitInBounds(rightCoarseBounds,coarseBounds)
        leftCropped = torchCrop(originalImage,leftBounds)
        rightCropped = torchCrop(originalImage,rightBounds)
        LRFullSizeMask = torch.where(LRFullSizeMask > 0,1,0)
        LRMaskDilated = pytorchBinaryDilation(LRFullSizeMask)



        # Get and post-process left lobe model output
        leftInput = torchPrep(leftCropped)
        leftLobeOutput = torchGetModelOutput(leftInput,self.leftModel)
        leftLobeOutput = torch.nn.functional.interpolate(leftLobeOutput, size=leftCropped.shape[2:], mode='nearest')
        # Get and post-process right lobe model output
        rightInput = torchPrep(rightCropped)
        rightLobeOutput = torchGetModelOutput(rightInput,self.rightModel)       
        rightLobeOutput = torch.nn.functional.interpolate(rightLobeOutput, size=rightCropped.shape[2:], mode='nearest')


        #adjust right lobe output (0,1,2,3) to (0,3,4,5)
        rightLobeOutput = rightLobeOutput + 2
        rightLobeOutput = torch.where(rightLobeOutput==2,0,rightLobeOutput)
        
        #Assemble final mask
        finalMask = torch.zeros(originalImage.shape).cuda()
        leftFullSize = torch.zeros(originalImage.shape).cuda()
        rightFullSize = torch.zeros(originalImage.shape).cuda()
        leftFullSize[:,:,leftBounds[0]:leftBounds[1],leftBounds[2]:leftBounds[3],leftBounds[4]:leftBounds[5]] = leftLobeOutput
        rightFullSize[:,:,rightBounds[0]:rightBounds[1],rightBounds[2]:rightBounds[3],rightBounds[4]:rightBounds[5]] = rightLobeOutput
        finalMask = torch.where(leftFullSize > 0, leftFullSize, finalMask)
        finalMask = torch.where(rightFullSize > 0, rightFullSize, finalMask)

        finalMask = torch.where(LRMaskDilated > 0,finalMask,0)

        finalMask = finalMask.to(torch.uint8)

        if getLR == 2:
            LRFinalMask = torch.where(torch.logical_or(LRFullSizeMask==1,LRFullSizeMask==2),1,0)
            LRFinalMask = torch.where(torch.logical_or(LRFinalMask==3,LRFinalMask==4,LRFinalMask==5),2,LRFinalMask)
            return LRFinalMask

        return finalMask
                




