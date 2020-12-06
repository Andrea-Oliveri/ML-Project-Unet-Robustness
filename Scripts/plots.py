# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from utils import get_binary_predictions


def plot_history(history):
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
    """Show.
    
    Args:
        image::[np.array]
            The output measures associated to the input measures tX.
        mask::[np.array]
            The input measures.
            
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
    
    
def plot_results(results, parameter_name, model_keys):
    """
    
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
    for key in model_keys:        
        line, = plt.plot(results_models[key]["recalls"], results_models[key]["precisions"]    , label=key.capitalize())
        plt.scatter(results_models[key]["recalls"][0]  , results_models[key]["precisions"][0] , color=line.get_color(), marker='o')
        plt.scatter(results_models[key]["recalls"][-1] , results_models[key]["precisions"][-1], color=line.get_color(), marker='o')
        plt.annotate(parameter[0] , (results_models[key]["recalls"][0] , results_models[key]["precisions"][0]) , fontsize=12)
        plt.annotate(parameter[-1], (results_models[key]["recalls"][-1], results_models[key]["precisions"][-1]), fontsize=12)
    plt.legend()
    plt.grid()
    
    
def read_results(results, model_keys):
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