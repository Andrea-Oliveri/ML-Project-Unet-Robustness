# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def get_dataset_from_folders(images_path, masks_path, images_shape=None, n_patches_per_image=6):
    images = read_images_from_folder(images_path)
    masks = read_images_from_folder(masks_path)
    
    if images_shape is None:
        return np.array(images), np.array(masks)
    
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


def normalize(img, nb_bits=1):
    """Par default, normalise entre 0 et 1, mais si nb_bits>1 specified, it scales it to max representable 
       by an unsigned of length nb_bits
       
    """
    min_val = np.min(img)
    max_val = np.max(img)
    return (2**nb_bits-1)*(img-min_val)/(max_val-min_val)


def get_binary_predictions(images, model):
    pred = model.predict(images, batch_size=1)
    return np.rint(pred).astype(np.uint8)

          
def get_number_cells(images, total=True):
    n_cells_images = []
    for image in images:
        n_cells, _ = cv2.connectedComponents(image)
        n_cells_images.append(n_cells)
       
    if total:
        return sum(n_cells_images)
        
    return np.array(n_cells_images)


def compute_jaccard_score(predictions, masks):
    n_images = predictions.shape[0]
    jaccard_images = np.zeros(n_images)
    
    for i in range(n_images):
        numerator = np.sum(np.logical_and(predictions[i], masks[i]))
        denominator = np.sum(np.logical_or(predictions[i], masks[i]))
        if denominator: 
            jaccard_images[i] = numerator/denominator
            
    return np.mean(jaccard_images)


def compute_precision_recall(predictions, masks):
    n_images = predictions.shape[0]
    precision = np.zeros(n_images)
    recall = np.zeros(n_images)
    
    for i in range(n_images):
        true_positives  = np.sum(np.logical_and(predictions[i]==1, masks[i]==1))
        false_positives = np.sum(np.logical_and(predictions[i]==1, masks[i]==0))
        false_negatives = np.sum(np.logical_and(predictions[i]==0, masks[i]==1))        
        
        denominator_precision = true_positives + false_positives
        if denominator_precision:
            precision[i] = true_positives / denominator_precision
        
        denominator_recall = true_positives + false_negatives
        if denominator_recall:
            recall[i] = true_positives / denominator_recall
         
    return np.mean(precision), np.mean(recall)