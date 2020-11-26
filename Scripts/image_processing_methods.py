# -*- coding: utf-8 -*-
import numpy as np


def normalizeExpansion(img):
    """
    

    """
    # Declare the output image
    output = np.copy(img)
    
    output = (img-np.min(img))/(np.max(img)-np.min(img))
    
    # Return the output image
    return output

def dog(image, sigma_1):
    """
    

    """
    output = np.copy(image)
    #gauss1 = skimage.filters.gaussian(output, sigma = sigma_1, preserve_range = True)
    
    sigma_2 = np.sqrt(2)*sigma_1
    #gauss2 = skimage.filters.gaussian(output, sigma = sigma_2, preserve_range = True)
    
    #output = gauss1 - gauss2
    output = normalizeExpansion(output)
    
    return output
