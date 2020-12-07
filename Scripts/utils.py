# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np


def get_dataset_from_folders(images_path, masks_path, images_shape=None, n_patches_per_image=6):
    """
    Read all images and their corresponding masks from folder paths given as parameters. The shape of the outputed images
    can be precised, and if so patches of the right size are taken from the image. The number of patches can per image can 
    also be given as parameter.

    Args:
        images_path::[str]
            The path to find the folder containing the images.
        masks_path::[str]
            The path to find the folder containing the masks corresponding to the images.
        images_shape::[tuple]
            The shape of the images and masks we want as output.
        n_patches_per_image::[int]
            The number of patches we want to retrieve from the read images.
    Returns:
        images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the images in the folder images_path.
        masks::[np.array]
            Numpy array of same shape as images containing the masks in the folder masks_path.
            
    """
    
    images = read_images_from_folder(images_path)
    masks = read_images_from_folder(masks_path)
    
    if images_shape is None:
        return np.array(images), np.array(masks)
    
    return split_images_and_masks_into_patches(images, masks, images_shape, n_patches_per_image)


def read_images_from_folder(path, extension=".tif"):
    """
    Read the images in the folder given by its path and given the extension of the images we want to read.
    
    Args:
        path::[str]
            The path to find the folder containing the images we want to read.
        extension::[str]
            The extension of the images we want to read.
    Returns:
        images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the images in the folder path.
            
    """
    images_filenames = sorted([name for name in os.listdir(path) if name.endswith(extension)])
    images = [cv2.imread(path + filename, cv2.IMREAD_ANYDEPTH) for filename in images_filenames]
    return images


def split_images_and_masks_into_patches(images, masks, patch_shape=(256, 256, 1), n_patches=6):
    """
    Split the given images and masks in n_patches patches, each patches being of shape patch_shape.
    
    Args:
        images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the images. 
        masks::[np.array]
            Numpy array of same shape as images containing the masks.
        patch_shape::[tuple]
            The shape of the images and masks we want as output.
        n_patches::[int]
            The number of patches we want to retrieve from the original images and masks.
    Returns:
        patches_images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the patches of images retrieved from the
            original images.
        patches_masks::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the patches of masks retrieved from the
            original masks.
        
    """
    patches_images = []
    patches_masks = []
    patch_lines, patch_cols, _ = patch_shape

    for image, mask in zip(images, masks):
        lines, cols = image.shape
        for _ in range(n_patches):
            random_line = np.random.choice(lines-patch_lines)
            random_col  = np.random.choice(cols-patch_cols)
            patch_image = np.reshape(image[random_line:random_line+patch_lines, random_col:random_col+patch_cols], patch_shape)
            patch_mask  = np.reshape(mask [random_line:random_line+patch_lines, random_col:random_col+patch_cols], patch_shape)
            patches_images.append(patch_image)
            patches_masks.append(patch_mask)
    
    return np.array(patches_images), np.array(patches_masks)


def normalize(image, nb_bits=1):
    """
    Normalize the image on a given number of bits n_bits.

    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        nb_bits::[int]
            The number of bits by which we want to normalize. By default it normalizes the
            image to be in the range [0,1].
    Returns:
        normalized_image::[np.array]
            Numpy array containing the normalized version of the image.
       
    """
    min_val = np.min(image)
    max_val = np.max(image)
    return (2**nb_bits-1)*(image-min_val)/(max_val-min_val)


def get_binary_predictions(images, model):
    """
    Uses the model passed as parameter to predict the segmentation masks of images and then uses a threshold and then 
    binarizes them.
    
    Args:
        images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the images to segment.
        model::[object]
            An object of a class implementing a predict method taking as parameters images and batch_size, and
            returning a prediction for the segmentation masks where each pixel is in range [0,1].
            
    Returns:
        pred::[np.array]
            Numpy array of same shape as images and containing the predicted binarized masks of the images.

    """
    pred = model.predict(images, batch_size=1)
    return np.rint(pred).astype(np.uint8)

          
def get_number_cells(images, total=True):
    """
    Retrieve the number of cells for each images by using a connected component techniques. If total is True then the sum of 
    all cells in the images is outputed otherwise a np.array containing the number of cells for each images is outputed.
    
    Args:
        images::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the images from which we want to count 
            the cells.
        total::[bool]
            If total is True then the sum of all cells in the images is outputed otherwise a np.array containing the number 
            of cells for each images is outputed.
    Returns:
        n_cells_images::[int] or [np.array]
            If total is True then the sum of all cells in the images is outputed otherwise a np.array of shape (n_images)
            containing the number of cells for each images is outputed.

    """
    n_cells_images = []
    for image in images:
        n_cells, _ = cv2.connectedComponents(image)
        n_cells_images.append(n_cells)
       
    if total:
        return sum(n_cells_images)
        
    return np.array(n_cells_images)


def compute_jaccard_score(predictions, masks):
    """
    Compute the mean Jaccard score over the predictions and true masks given in the parameters.
    
    Args:
        predictions::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the binary output predicted by the model.
        masks::[np.array]
            Numpy array of same shape as predictions containing the binary masks.
    Returns:
        jaccard_images::[float]
            The computed mean Jaccard score for the given predictions and true masks.

    """
    n_images = predictions.shape[0]
    jaccard_images = np.zeros(n_images)
    
    for i in range(n_images):
        numerator = np.sum(np.logical_and(predictions[i], masks[i]))
        denominator = np.sum(np.logical_or(predictions[i], masks[i]))
        if denominator: 
            jaccard_images[i] = numerator/denominator
            
    return np.mean(jaccard_images)


def compute_precision_recall(predictions, masks):
    """
    Compute the mean precision and the mean recall over the predictions and true masks given in the parameters..
    
    Args:
        predictions::[np.array]
            Numpy array of shape (n_images, n_lines, n_columns, n_channels) containing the binary output predicted by the model.
        masks::[np.array]
            Numpy array of same shape as predictions containing the binary masks.
    Returns:
        precision::[float]
            The computed mean precision for the given predictions and true masks.
        recall::[float]
            The computed mean recall for the given predictions and true masks.
    """
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
            
            
def read_results(results, model_keys):
    """
    Reads the results dictionary passed as parameter and converts it into lists.
    
    Args:
        results::[dict]
            Dictionary of form {parameter_value: {model_name: {metric_name: metric_val, ...}, ...}, ...}
        model_keys::[dict_keys]
            Variable collecting the names of the models as used in the keys the results child dictionaries.
    
    Returns:
        parameter::[list]
            List of parameter's values.
        results_models::[dictionary]
            Dictionary of form {model_name: {metric_name: metric_vals_list}, ...}
    """
    parameter          = []
    number_cells_masks = []
    results_models     = {}
    
    for key in model_keys:
        results_models[key] = {"accuracies": [], "jaccards": [], "precisions": [], "recalls": [], "number_cells_predictions": []}
 
    for parameter_value, measures in results.items():
        parameter.append(parameter_value)
        number_cells_masks.append(measures[list(model_keys)[0]]["number_cells_masks"])

        for key in model_keys:
            results_models[key]["accuracies"].append( measures[key]["accuracy"] )
            results_models[key]["jaccards"]  .append( measures[key]["jaccard"] )
            results_models[key]["precisions"].append( measures[key]["precision"] )
            results_models[key]["recalls"]   .append( measures[key]["recall"] )
            results_models[key]["number_cells_predictions"].append( measures[key]["number_cells_predictions"] )     
        
    return parameter, results_models, number_cells_masks