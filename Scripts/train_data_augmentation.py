# -*- coding: utf-8 -*-
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from distortions import add_gaussian, add_gaussian_noise


def make_train_generator(train_images, train_masks, batch_size=1, custom_distortions=False, seed=2):
    """
    
    """
    if custom_distortions:
        images_datagen = ImageDataGenerator(preprocessing_function=data_augmentation_with_distortions_for_images)
        masks_datagen  = ImageDataGenerator(preprocessing_function=data_augmentation_with_distortions_for_masks)
    else:
        images_datagen = ImageDataGenerator(preprocessing_function=data_augmentation_no_distortions)
        masks_datagen  = ImageDataGenerator(preprocessing_function=data_augmentation_no_distortions)

    # Provide the same seed and keyword arguments to the flow method so that images and masks transformed the same way.
    images_generator = images_datagen.flow(train_images, batch_size=batch_size, seed=seed)
    masks_generator  = masks_datagen .flow(train_masks , batch_size=batch_size, seed=seed)

    # Combine generators into one which yields image and masks.
    train_generator = zip(images_generator, masks_generator)

    return train_generator



def data_augmentation_no_distortions(image):
    """
    
    """
    random_rotation    = np.random.choice([0,1,2,3])
    possible_flip_axis = np.array([(), (1), (2), (1,2)], dtype=object)
    random_flip_axis   = np.random.choice(possible_flip_axis)

    new_image = np.rot90(image, random_rotation)
    new_image = np.flip(image, random_flip_axis)
    
    return new_image


def data_augmentation_with_distortions_for_images(image):
    """
    
    """
    new_image = data_augmentation_no_distortions(image)

    new_image = apply_random_distortion_from_range(add_gaussian, new_image, {"amplitude": (0, 300)})

    new_image = apply_random_distortion_from_range(add_gaussian_noise, new_image, {"mean": (0, 0), "sigma": (0, 30)})

    return new_image


def data_augmentation_with_distortions_for_masks(mask):
    """
    
    """
    new_image = data_augmentation_no_distortions(mask)
    
    return new_image


def apply_random_distortion_from_range(function, image, params_ranges={}):
    random_params = {}
    for param, val_range in params_ranges.items():
        random_params[param] = np.random.uniform(*val_range)

    distorted_image = function(image, **random_params)
    
    return distorted_image
