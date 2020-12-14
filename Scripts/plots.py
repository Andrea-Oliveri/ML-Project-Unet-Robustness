# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from utils import get_binary_predictions, read_results


def plot_history(history):
    """
    Plots the evolution of accuracy and loss over the epochs as collected in history dictionary.
    
    Args:
        history::[dict]
            Dictionary of form {metric: metric_val_list, ...}
            
    """
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'])
    plt.show()
    
    
def show_image_mask(image, mask):
    """
    Shows the image and true mask in a row of subplots.
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        mask::[np.array]
            Numpy array of same shape as image containing its true mask. 
                   
    """
    fig, axes = plt.subplots(1, 2)
    ax_image, ax_mask = axes
    ax_image.imshow(image, 'gray')
    ax_image.axis('off')
    ax_image.set_title("Original Image")
    ax_mask.imshow(mask, 'gray')
    ax_mask.axis('off')
    ax_mask.set_title("Original Mask")
    plt.show()
    
    
def show_image_pred(image, models):
    """
    Shows the image and predictions of each model for that image in a row of subplots.
    
    Args:
        image::[np.array]
            Numpy array containing one image of shape (n_lines, n_columns, n_channels).
        models::[dict]
            Dictionary of form {model_name: model_object, ...}
                   
    """
    n_models = len(models)
    
    fig, axes = plt.subplots(1, n_models+1, figsize=(16, 4.8))
    
    ax_image, *ax_preds = axes
    
    ax_image.imshow(image, 'gray')
    ax_image.axis('off')
    ax_image.set_title("Image used for Predictions")
    
    for idx, (model_key, model) in enumerate(models.items()):
        pred = get_binary_predictions(image[None, :], model)
        ax_preds[idx].imshow(pred.squeeze(), 'gray')
        ax_preds[idx].axis('off')
        ax_preds[idx].set_title(f"Prediction by\n{model_key.capitalize()}")
    
    plt.show()
    
    
def plot_results(results, model_keys, parameter_name, parameter_values_to_annotate=[]):
    """
    Plots the accuracy, jaccard score, number of cells detected and precision-recall curves for each model as collected
    in the results argument. 
    
    Args:
        results::[dict]
            Dictionary of form {parameter_value: {model_name: {metric_name: metric_val, ...}, ...}, ...}
        model_keys::[dict_keys]
            Variable collecting the names of the models as used in the keys the results child dictionaries.
        parameter_name::[string]
            The name of the parameter that was being tested when collecting result. Only needed to label the x-axis.
        parameter_values_to_annotate::[list]
            List of points corresponding to parameter values to be annotated in precision-recall plot.
            
    """
    parameter, results_models, number_cells_masks = read_results(results, model_keys)
    
    plt.figure()
    plt.title("Accuracy")
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    for key in model_keys:
        plt.plot(parameter, results_models[key]["accuracies"], label=key.capitalize())
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Jaccard Score")
    plt.xlabel(parameter_name)
    plt.ylabel("Jaccard Score")
    for key in model_keys:
        plt.plot(parameter, results_models[key]["jaccards"], label=key.capitalize())
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Number of Cells Detected")
    plt.xlabel(parameter_name)
    plt.ylabel("Number of Cells Detected")
    for key in model_keys:
        plt.plot(parameter, results_models[key]["number_cells_predictions"], label=key.capitalize())
    plt.plot(parameter, number_cells_masks, linestyle='--', label="Masks", )    
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim((0,1))
    plt.ylim((0,1))
    annotated_points_idx = np.unique([np.abs(np.array(parameter)-value).argmin() for value in parameter_values_to_annotate])
    for key in model_keys:
        line, = plt.plot(results_models[key]["recalls"], results_models[key]["precisions"], label=key.capitalize())
        for idx in annotated_points_idx:
            plt.scatter(results_models[key]["recalls"][idx], results_models[key]["precisions"][idx], color=line.get_color(),
                        marker='x')
            plt.annotate(parameter[idx], (results_models[key]["recalls"][idx], results_models[key]["precisions"][idx]),
                         fontsize=10)
    plt.legend()
    plt.grid()