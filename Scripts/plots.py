# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
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
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 4.8))
    
    ax_image, ax_pred_original, ax_pred_retrained = axes
    ax_image.imshow(image, 'gray')
    ax_image.axis('off')
    ax_image.set_title("Image used for Predictions")
    
    original_pred = get_binary_predictions(image[None, :], models["original"])
    ax_pred_original.imshow(original_pred.squeeze(), 'gray')
    ax_pred_original.axis('off')
    ax_pred_original.set_title("Prediction by Original UNET")
    
    retrained_pred = get_binary_predictions(image[None, :], models["retrained"])
    ax_pred_retrained.imshow(retrained_pred.squeeze(), 'gray')
    ax_pred_retrained.axis('off')
    ax_pred_retrained.set_title("Prediction by Retrained UNET")
    
    plt.show()
    
    
def plot_results(results, parameter_name):
    """
    
    """
    parameter, results_original, results_retrained, number_cells_masks = read_results(results)

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    plt.plot(parameter, results_original ["accuracies"], label="Original Model")
    plt.plot(parameter, results_retrained["accuracies"], label="Retrained Model")
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Jaccard Score")
    plt.xlabel(parameter_name)
    plt.ylabel("Jaccard Score")
    plt.plot(parameter, results_original ["jaccards"], label="Original Model")
    plt.plot(parameter, results_retrained["jaccards"], label="Retrained Model")
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Number of Cells Detected")
    plt.xlabel(parameter_name)
    plt.ylabel("Number of Cells Detected")
    plt.plot(parameter, results_original ["number_cells_predictions"], label="Original Model")
    plt.plot(parameter, results_retrained["number_cells_predictions"], label="Retrained Model")
    plt.plot(parameter, number_cells_masks, linestyle='--'           , label="Masks", )    
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim((0,1))
    plt.ylim((0,1))
    
    plt.plot(   results_original["recalls"]    , results_original["precisions"]    , label="Original Model")
    plt.scatter(results_original["recalls"][0] , results_original["precisions"][0] , marker='o')
    plt.scatter(results_original["recalls"][-1], results_original["precisions"][-1], marker='o')
    plt.annotate(parameter[0] , (results_original["recalls"][0] , results_original["precisions"][0]) , fontsize=12)
    plt.annotate(parameter[-1], (results_original["recalls"][-1], results_original["precisions"][-1]), fontsize=12)
    
    plt.plot(   results_retrained["recalls"]    , results_retrained["precisions"]    , label="Retrained Model")
    plt.scatter(results_retrained["recalls"][0] , results_retrained["precisions"][0] , marker='o')
    plt.scatter(results_retrained["recalls"][-1], results_retrained["precisions"][-1], marker='o')
    plt.annotate(parameter[0] , (results_retrained["recalls"][0] , results_retrained["precisions"][0]) , fontsize=12)
    plt.annotate(parameter[-1], (results_retrained["recalls"][-1], results_retrained["precisions"][-1]), fontsize=12)
    
    plt.legend()
    plt.grid()
    
    
def read_results(results):
    parameter          = []
    results_original   = {"accuracies": [], "jaccards": [], "precisions": [], "recalls": [], "number_cells_predictions": []}
    results_retrained  = {"accuracies": [], "jaccards": [], "precisions": [], "recalls": [], "number_cells_predictions": []}
    number_cells_masks = []
     
    for parameter_value, measures in results.items():
        parameter.append(parameter_value)
        number_cells_masks.append(measures["original"]["number_cells_masks"])

        results_original ["accuracies"].append(measures["original"] ["accuracy"])
        results_retrained["accuracies"].append(measures["retrained"]["accuracy"])
        results_original ["jaccards"]  .append(measures["original"] ["jaccard"])
        results_retrained["jaccards"]  .append(measures["retrained"]["jaccard"])
        results_original ["precisions"].append(measures["original"] ["precision"])
        results_retrained["precisions"].append(measures["retrained"]["precision"])
        results_original ["recalls"]   .append(measures["original"] ["recall"])
        results_retrained["recalls"]   .append(measures["retrained"]["recall"])
        results_original ["number_cells_predictions"].append(measures["original"] ["number_cells_predictions"])
        results_retrained["number_cells_predictions"].append(measures["retrained"]["number_cells_predictions"])
        
        
    return parameter, results_original, results_retrained, number_cells_masks