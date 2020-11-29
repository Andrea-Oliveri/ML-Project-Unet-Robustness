# -*- coding: utf-8 -*-
import numpy as np
import cv2


def normalize(img, nb_bits=1):
    """Par default, normalise entre 0 et 1, mais si nb_bits>1 specified, it scales it to max representable 
       by an unsigned of length nb_bits
       
    """
    min_val = np.min(img)
    max_val = np.max(img)
    return (2**nb_bits-1)*(img-min_val)/(max_val-min_val)


def dog(image, sigma_low, sigma_high=None):
    """
    

    """
    if sigma_high is None:
        sigma_high = np.sqrt(2)*sigma_low
        
    gauss_low = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=sigma_low, sigmaY=sigma_low)
    gauss_high = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=sigma_high, sigmaY=sigma_high)
    
    output = gauss_low - gauss_high
    
    return normalize(output, nb_bits=8)
