# -*- coding: utf-8 -*-
import numpy as np
import cv2
from abc import ABC, abstractmethod
from utils import normalize


class ImageProcessingMethod(ABC):
    """Class ImageProcessingMethod. Virtual class representing a classical image processing method used for segmentation."""        

    @abstractmethod
    def apply_filter(self, image):
        """ """
        pass

    
    def predict(self, images, batch_size=1):
        """Batch size is an unused compatibility parameter."""
        predictions = []
        
        for image in images:
            filtered_image = self.apply_filter(image)
            _, pred = cv2.threshold(filtered_image, 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            predictions.append(pred)
            
        return np.reshape(predictions, images.shape)
    
    
    
class DogImageProcessingMethod(ImageProcessingMethod):
    """Class DogImageProcessingMethod. Child class of ImageProcessingMethod using a Difference of Gaussian filter."""
    
    def __init__(self, ksize_low, ksize_high=None):
        """ """  
        self._sigma_low  = 0.3*(ksize_low//2 - 1) + 0.8
            
        if ksize_high is None:
            self._sigma_high = np.sqrt(2)*self._sigma_low
        else:
            self._sigma_high = 0.3*(ksize_high//2 - 1) + 0.8
        
    
    def apply_filter(self, image):
        """


        """
        gauss_low  = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=self._sigma_low , sigmaY=self._sigma_low)
        gauss_high = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=self._sigma_high, sigmaY=self._sigma_high)

        output = gauss_low - gauss_high

        return normalize(output, nb_bits=8)
    
    

class DenoiserImageProcessingMethod(ImageProcessingMethod):
    """Class DenoiserImageProcessingMethod. Child class of ImageProcessingMethod using a state-of-the-art denoiser."""
    
    def __init__(self, denoiser_strength):
        """ """  
        self._strength = denoiser_strength
        
     
    def apply_filter(self, image):
        """


        """
        return cv2.fastNlMeansDenoising(image.astype('uint8'), h=self._strength)