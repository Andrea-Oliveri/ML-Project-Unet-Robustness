# -*- coding: utf-8 -*-
import numpy as np
import cv2
from abc import ABC, abstractmethod
from utils import normalize


class ImageProcessingMethod(ABC):
    """
    Class ImageProcessingMethod. Abstract class representing a classical image processing method
    used for segmentation.
    """        

    @abstractmethod
    def apply_filter(self, image):
        """
        Abstract method applying a filter on the image passed as parameter. 
        
        Args:
            image::[np.array]
                Numpy array containing one image of shape (n_lines, n_columns, n_channels).
            
        Returns:
            filtered_image::[np.array]
                Numpy array containing the filtered image of same shape as input image 
                normalized to be in range [0, 255].
                
        """
        pass

    
    def predict(self, images, batch_size=1):
        """
        Predicts a segmentation mask by applying a filter and an Otsu thresholder.

        Args:
            image::[np.array]
                Numpy array containing one image of shape (n_lines, n_columns, n_channels).
            batch_size::[np.array]
                Unused parameter. Needed for compatibility purposes.

        Returns:
            prediction::[np.array]
                Numpy array containing the predicted segmentation mask of shape
                (n_lines, n_columns, n_channels).
        """
        predictions = []
        
        for image in images.astype("float"):
            filtered_image = self.apply_filter(image)
            _, pred = cv2.threshold(filtered_image.astype('uint8'), 0, 1, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            predictions.append(pred)
            
        return np.reshape(predictions, images.shape)
    
    
    
class DogImageProcessingMethod(ImageProcessingMethod):
    """
    Class DogImageProcessingMethod. Child class of ImageProcessingMethod using a Difference of
    Gaussian filter.
    """
    
    def __init__(self, ksize_low, ksize_high=None):
        """
        Constructor of DogImageProcessingMethod. Initialises the variables describing the 
        Difference of Gaussian filter.
        
        Args:
            ksize_low::[float]
                Number describing the kernel size of the low-std gaussian. 
            ksize_high::[float]
                Number describing the kernel size of the high-std gaussian. 

        """  
        self._sigma_low  = 0.3*(ksize_low//2 - 1) + 0.8
            
        if ksize_high is None:
            self._sigma_high = np.sqrt(2)*self._sigma_low
        else:
            self._sigma_high = 0.3*(ksize_high//2 - 1) + 0.8
        
    
    def apply_filter(self, image):
        """
        Applies a difference of gaussian filter (band-pass) on the image.

        Args:
            image::[np.array]
                Numpy array containing one image of shape (n_lines, n_columns, n_channels).

        Returns:
            filtered_image::[np.array]
                Numpy array containing the filtered image of same shape as input image 
                normalized to be in range [0, 255].
        """
        gauss_low  = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=self._sigma_low , sigmaY=self._sigma_low)
        gauss_high = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=self._sigma_high, sigmaY=self._sigma_high)

        filtered_image = gauss_low - gauss_high

        return normalize(filtered_image, nb_bits=8)
    
    

class DenoiserImageProcessingMethod(ImageProcessingMethod):
    """Class DenoiserImageProcessingMethod. Child class of ImageProcessingMethod using a
    state-of-the-art denoiser."""
    
    def __init__(self, denoiser_strength):
        """
        Constructor of DenoiserImageProcessingMethod. Initialises the variables describing the 
        Denoising filter strength.
        
        Args:
            denoiser_strength::[float]
                Number describing the denoiser filter strength (larger is stronger).
        """ 
        self._strength = denoiser_strength
        
     
    def apply_filter(self, image):
        """
        Applies a white gaussian noise denoiser (low-pass) on the image.

        Args:
            image::[np.array]
                Numpy array containing one image of shape (n_lines, n_columns, n_channels).

        Returns:
            filtered_image::[np.array]
                Numpy array containing the filtered image of same shape as input image 
                normalized to be in range [0, 255].
        """
        return cv2.fastNlMeansDenoising(image.astype('uint8'), h=self._strength)