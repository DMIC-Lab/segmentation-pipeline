import torch
from monai.networks.nets import SwinUNETR



from segpipe.pipelineComponents import dust, getModelOutput,binaryErosion,bbox_3D,prep,binaryDilation,crop,fallback
import numpy as np
from pkg_resources import resource_filename

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
    
    def segment(self,originalImage, getLR = 0, numpyDust = False):
        originalType = None
        if isinstance(originalImage, np.ndarray):
            originalImage = torch.from_numpy(originalImage).float()
            originalType = 'np'
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
        im = torch.sum(originalImage,dim=2).unsqueeze(2)
        im = im.repeat(1,1,originalImage.shape[2],1,1)
        im = torch.nn.functional.interpolate(im,size=(128,128,128),mode='nearest')
        eroded = binaryErosion((im/torch.max(im)) > .2,selem_radius=3)
        dilated = binaryDilation(eroded,selem_radius=3)
        dilated = torch.nn.functional.interpolate(dilated.to(torch.uint8),size=(originalImage.shape[2],originalImage.shape[3],originalImage.shape[4]),mode='nearest')
        originalImage = torch.where(dilated > 0, originalImage, torch.zeros_like(originalImage))

        lrImage = torch.nn.functional.interpolate(originalImage,size=(128,128,128),mode='nearest')
        
        #Segment LR - LR model outputs mask of what is left and right lung
        LRInput = prep(lrImage) #HWD -> NCHWD
        LROutput = getModelOutput(LRInput,self.LRModel)
        if numpyDust:
            LROutput = fallback(LROutput,device = self.device,threshold=5000)
        else:
            LROutput = dust(LROutput,device = self.device,threshold=5000)
        LROutput = torch.nn.functional.interpolate(LROutput, size=originalImage.shape[2:], mode='nearest')
        

        if getLR == 1:
            if originalType == 'np':
                return LROutput.squeeze(0).squeeze(0).cpu().numpy()
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
            leftBounds = bbox_3D(leftOutput,margin=5)
            leftCropped = crop(originalImage,leftBounds)
        else:
            leftCropped = None
        if rightOutput is not None:
            rightBounds = bbox_3D(rightOutput,margin=5)
            rightCropped = crop(originalImage,rightBounds)
        else:
            rightCropped = None



        if leftCropped is not None:
            # Get and post-process left lobe model output
            leftInput = prep(leftCropped)
            leftLobeOutput = getModelOutput(leftInput,self.leftModel)
            leftLobeOutput = torch.nn.functional.interpolate(leftLobeOutput, size=leftCropped.shape[2:], mode='nearest')
            temp = torch.zeros_like(leftLobeOutput)
            temp[:,:,:-1,:-1,:-1] = leftLobeOutput[:,:,1:,1:,1:]
            leftLobeOutput = temp
        if rightCropped is not None:
            # Get and post-process right lobe model output
            rightInput = prep(rightCropped)
            rightLobeOutput = getModelOutput(rightInput,self.rightModel)       
            rightLobeOutput = torch.nn.functional.interpolate(rightLobeOutput, size=rightCropped.shape[2:], mode='nearest')

            #adjust right lobe output (0,1,2,3) to (0,3,4,5)
            rightLobeOutput = rightLobeOutput + 2
            rightLobeOutput = torch.where(rightLobeOutput==2,0,rightLobeOutput)
            temp = torch.zeros_like(rightLobeOutput)
            temp[:,:,:-1,:-1,:-1] = rightLobeOutput[:,:,1:,1:,1:]
            rightLobeOutput = temp
        

        #Assemble final mask
        finalMask = torch.zeros(originalImage.shape).to(self.device)
        leftFullSize = torch.zeros(originalImage.shape).to(self.device)
        rightFullSize = torch.zeros(originalImage.shape).to(self.device)
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
        if numpyDust:
            finalMask = fallback(finalMask,device=self.device)
        else:
            finalMask = dust(finalMask,device=self.device)

        finalMask = torch.nn.functional.interpolate(finalMask, size=originalImage.shape[2:], mode='nearest-exact')
        
        finalMask = finalMask.to(torch.uint8)
        
        if originalType == 'np':
            finalMask = finalMask.squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)

        if getLR == 2:
            finalMask = np.where((finalMask == 1)|(finalMask == 2),1,finalMask)
            finalMask = np.where((finalMask == 3)|(finalMask == 4)|(finalMask == 5),2,finalMask)

        return finalMask