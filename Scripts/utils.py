# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import distance


def get_dataset_from_folders(images_path, masks_path, images_shape, n_patches_per_image=6):
    images = read_images_from_folder(images_path)
    masks = read_images_from_folder(masks_path)
    return split_images_and_masks_into_patches(images, masks, images_shape, n_patches_per_image)


def read_images_from_folder(path, extension=".tif"):
    images_filenames = sorted([name for name in os.listdir(path) if name.endswith(extension)])
    images = [cv2.imread(path + filename, cv2.IMREAD_ANYDEPTH) for filename in images_filenames]
    return images


def split_images_and_masks_into_patches(images, masks, patch_shape=(256, 256, 1), n_patches=6):
    patches_images = []
    patches_masks = []
    patch_lines, patch_cols, _ = patch_shape

    for image, mask in zip(images, masks):
        lines, cols = image.shape
        for _ in range(n_patches):
            random_line = np.random.choice(lines-patch_lines)
            random_col = np.random.choice(cols-patch_cols)
            patch_image = np.reshape(image[random_line:random_line+patch_lines, random_col:random_col+patch_cols], patch_shape)
            patch_mask = np.reshape(mask[random_line:random_line+patch_lines, random_col:random_col+patch_cols], patch_shape)
            patches_images.append(patch_image)
            patches_masks.append(patch_mask)
    
    return np.array(patches_images), np.array(patches_masks)


def get_binary_predictions(images, model):
    pred = model.predict(images, batch_size=1)
    return np.rint(pred).astype(np.uint8)
          
def get_number_cells(images, total=True):
    n_cells_images = []
    for image in images:
        _, n_cells = ndimage.label(image)
        n_cells_images.append(n_cells)
       
    if total:
        n_cells_images = sum(n_cells_images)
    
    return n_cells_images

def compute_jaccard_index(im1, im2):
    return distance.jaccard(im1, im2)
