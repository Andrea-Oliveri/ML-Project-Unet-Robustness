# -*- coding: utf-8 -*-
import cv2
import numpy as np


def generate_gaussian2D(img_shape, sigma_x, sigma_y, mu_x, mu_y):
    """
    
    """
    pixels_range = range(-img_shape[0]//2, img_shape[1]//2)
    x, y = np.meshgrid(pixels_range, pixels_range)
    gaussian_x = np.exp(- (x - mu_x)**2 / (2 * sigma_x**2))
    gaussian_y = np.exp(- (y - mu_y)**2 / (2 * sigma_y**2))
    gaussian = gaussian_x * gaussian_y
    
    return gaussian.reshape(img_shape)


def normalize_bits(img, nb_bits=1):
    """
    Par default, normalise entre 0 et 1, mais si nb_bits>1 specified, it scales it to max representable by an unsigned of length nb_bits
    """
    return (2**nb_bits - 1)*(img-np.min(img))/(np.max(img)-np.min(img))


def add_gaussian(img, amplitude, sigmaX=None, sigmaY=None, mu_x=None, mu_y=None, nb_bits=8):
    """

    """
    if(sigmaX == None):
        sigmaX = (img.shape[0])/5
    if(sigmaY == None):
        sigmaY = (img.shape[1])/5
    if(mu_x == None):
        mu_x = np.random.randint(-img.shape[0] // 2, img.shape[0] // 2)
    if(mu_y == None):
        mu_y = np.random.randint(-img.shape[1] // 2, img.shape[1] // 2)
        
    gaussian = generate_gaussian2D(img.shape, sigmaX, sigmaY, mu_x, mu_y)
    new_img = gaussian * amplitude + img
    
    return normalize_bits(new_img, nb_bits)


def zoom_image(img, zoom_factor, val_padding=None):
    """

    """
    if zoom_factor <= 0:
        raise ValueError("The zoom factor must be strictly positive")
    
    height, width, _ = img.shape
    width_zoom = int(width * zoom_factor)
    height_zoom = int(height * zoom_factor)
    
    diff_width = width_zoom - width
    diff_height = height_zoom - height
    
    if zoom_factor == 1:
        return img
    
    elif zoom_factor > 1:
        # Crop image
        x1 = diff_width // 2
        y1 = diff_height // 2
        
        x2 = x1 + width
        y2 = y1 + height
        
        edges = np.array([x1,y1,x2,y2])
        edges = edges / zoom_factor
        x1, y1, x2, y2 = edges.astype(np.int)
        cropped_img = img[y1:y2, x1:x2]
        
        # Resize
        output = cv2.resize(cropped_img, (width, height), interpolation=cv2.INTER_CUBIC)
        
    elif zoom_factor < 1:
        # Resize
        output = cv2.resize(img, (width_zoom, height_zoom), interpolation=cv2.INTER_CUBIC)
        
        if val_padding == None:
            val_padding = img.min()
        
        # Padding with val_padding
        x1 = -diff_width // 2
        y1 = -diff_height // 2
        
        x2 = -diff_width - x1
        y2 = -diff_height - y1
        
        pad_width = [(y1, y2), (x1, x2)]
        
        output = np.pad(output, pad_width, 'constant', constant_values = val_padding)

    return output.reshape(img.shape)


def add_gaussian_noise(img, mean, std, nb_bits=8):
    gaussian = np.random.normal(mean, std, img.shape)
    output = normalize_bits(img + gaussian, nb_bits)
    return output