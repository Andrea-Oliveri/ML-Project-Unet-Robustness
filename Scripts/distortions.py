# -*- coding: utf-8 -*-
import cv2
import numpy as np
from image_processing_methods import normalize


def generate_gaussian2D(img_shape, sigma_x, sigma_y, mu_x, mu_y):
    """
    
    """
    pixels_range = range(-img_shape[0]//2, img_shape[1]//2)
    x, y = np.meshgrid(pixels_range, pixels_range)
    gaussian_x = np.exp(- (x - mu_x)**2 / (2 * sigma_x**2))
    gaussian_y = np.exp(- (y - mu_y)**2 / (2 * sigma_y**2))
    gaussian = gaussian_x * gaussian_y
    
    return gaussian.reshape(img_shape)


def add_gaussian(img, amplitude, sigmaX=None, sigmaY=None, mu_x=None, mu_y=None, nb_bits=8):
    """

    """
    if sigmaX is None:
        sigmaX = (img.shape[0])/5
    if sigmaY is None:
        sigmaY = (img.shape[1])/5
    if mu_x is None:
        mu_x = np.random.randint(-img.shape[0] // 2, img.shape[0] // 2)
    if mu_y is None:
        mu_y = np.random.randint(-img.shape[1] // 2, img.shape[1] // 2)
        
    gaussian = generate_gaussian2D(img.shape, sigmaX, sigmaY, mu_x, mu_y)
    new_img = gaussian * amplitude + img
    
    return normalize(new_img, nb_bits)


def add_gaussian_noise(img, mean, sigma, nb_bits=8):
    gaussian = np.random.normal(mean, sigma, img.shape)
    output = normalize(img + gaussian, nb_bits)
    return output


def zoom_image(img, zoom_factor, val_padding=None):
    """

    """
    if zoom_factor <= 0:
        raise ValueError("The zoom factor must be strictly positive")
    
    elif zoom_factor == 1:
        output = img
        
    else:
        # Resize
        new_img = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_CUBIC)
        
        height, width, _      = img.shape
        new_height, new_width = new_img.shape
    
        if zoom_factor > 1:
            # Crop central portion of correct size
            line_start = (new_height - height) // 2
            col_start  = (new_width - width) // 2
            line_end   = line_start + height
            col_end    = col_start  + width

            output = new_img[line_start:line_end, col_start:col_end]

        elif zoom_factor < 1:  
            if val_padding is None:
                val_padding = img.min()

            # Padding with val_padding
            pad_up    = (height - new_height) // 2
            pad_down  = height - pad_up - new_height
            pad_left  = (width - new_width) // 2
            pad_right = width - pad_left - new_width

            output = np.pad(new_img, ((pad_up, pad_down), (pad_left, pad_right)), 'constant', constant_values = val_padding)

    return output.reshape(img.shape)


def zoom_image_to_meet_shape(img, shape):
    """

    """
    return cv2.resize(img, (shape[0], shape[1]), interpolation=cv2.INTER_CUBIC).reshape(shape)