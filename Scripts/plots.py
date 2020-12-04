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
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.8))
    
    ax_image, ax_pred_original, ax_pred_data_augmented, ax_pred_data_distorted = axes
    ax_image.imshow(image, 'gray')
    ax_image.axis('off')
    ax_image.set_title("Image used for Predictions")
    
    pred = get_binary_predictions(image[None, :], models["original"])
    ax_pred_original.imshow(pred.squeeze(), 'gray')
    ax_pred_original.axis('off')
    ax_pred_original.set_title("Prediction by Original UNET")
    
    pred = get_binary_predictions(image[None, :], models["data augmented"])
    ax_pred_data_augmented.imshow(pred.squeeze(), 'gray')
    ax_pred_data_augmented.axis('off')
    ax_pred_data_augmented.set_title("Prediction by Data-Augmented UNET")
    
    pred = get_binary_predictions(image[None, :], models["data distorted"])
    ax_pred_data_distorted.imshow(pred.squeeze(), 'gray')
    ax_pred_data_distorted.axis('off')
    ax_pred_data_distorted.set_title("Prediction by Data-Distorted UNET")
    
    plt.show()
    
    
def plot_results(results, parameter_name):
    """
    
    """
    parameter, results_original, results_data_augmented, results_data_distorted, number_cells_masks = read_results(results)

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    plt.plot(parameter, results_original      ["accuracies"], label="Original Model")
    plt.plot(parameter, results_data_augmented["accuracies"], label="Data-Augmented Model")
    plt.plot(parameter, results_data_distorted["accuracies"], label="Data-Distorted Model")
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Jaccard Score")
    plt.xlabel(parameter_name)
    plt.ylabel("Jaccard Score")
    plt.plot(parameter, results_original      ["jaccards"], label="Original Model")
    plt.plot(parameter, results_data_augmented["jaccards"], label="Data-Augmented Model")
    plt.plot(parameter, results_data_distorted["jaccards"], label="Data-Distorted Model")
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Number of Cells Detected")
    plt.xlabel(parameter_name)
    plt.ylabel("Number of Cells Detected")
    plt.plot(parameter, results_original      ["number_cells_predictions"], label="Original Model")
    plt.plot(parameter, results_data_augmented["number_cells_predictions"], label="Data-Augmented Model")
    plt.plot(parameter, results_data_distorted["number_cells_predictions"], label="Data-Distorted Model")
    plt.plot(parameter, number_cells_masks, linestyle='--'                , label="Masks", )    
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim((0,1))
    plt.ylim((0,1))
    
    plt.plot(   results_original["recalls"]    , results_original["precisions"]    , label ="Original Model")
    plt.scatter(results_original["recalls"][0] , results_original["precisions"][0] , marker='o')
    plt.scatter(results_original["recalls"][-1], results_original["precisions"][-1], marker='o')
    plt.annotate(parameter[0] , (results_original["recalls"][0] , results_original["precisions"][0]) , fontsize=12)
    plt.annotate(parameter[-1], (results_original["recalls"][-1], results_original["precisions"][-1]), fontsize=12)
    
    plt.plot(   results_data_augmented["recalls"]    , results_data_augmented["precisions"]    , label ="Data-Augmented Model")
    plt.scatter(results_data_augmented["recalls"][0] , results_data_augmented["precisions"][0] , marker='o')
    plt.scatter(results_data_augmented["recalls"][-1], results_data_augmented["precisions"][-1], marker='o')
    plt.annotate(parameter[0] , (results_data_augmented["recalls"][0] , results_data_augmented["precisions"][0]) , fontsize=12)
    plt.annotate(parameter[-1], (results_data_augmented["recalls"][-1], results_data_augmented["precisions"][-1]), fontsize=12)
    
    plt.plot(   results_data_distorted["recalls"]    , results_data_distorted["precisions"]    , label ="Data-Distorted Model")
    plt.scatter(results_data_distorted["recalls"][0] , results_data_distorted["precisions"][0] , marker='o')
    plt.scatter(results_data_distorted["recalls"][-1], results_data_distorted["precisions"][-1], marker='o')
    plt.annotate(parameter[0] , (results_data_distorted["recalls"][0] , results_data_distorted["precisions"][0]) , fontsize=12)
    plt.annotate(parameter[-1], (results_data_distorted["recalls"][-1], results_data_distorted["precisions"][-1]), fontsize=12)
    
    plt.legend()
    plt.grid()
    
    
def read_results(results):
    parameter               = []
    results_original        = {"accuracies": [], "jaccards": [], "precisions": [], "recalls": [], "number_cells_predictions": []}
    results_data_augmented  = {"accuracies": [], "jaccards": [], "precisions": [], "recalls": [], "number_cells_predictions": []}
    results_data_distorted  = {"accuracies": [], "jaccards": [], "precisions": [], "recalls": [], "number_cells_predictions": []}
    number_cells_masks      = []
     
    for parameter_value, measures in results.items():
        parameter.append(parameter_value)
        number_cells_masks.append(measures["original"]["number_cells_masks"])

        for model_key, results_per_model in {"original"      : results_original, 
                                             "data augmented": results_data_augmented, 
                                             "data distorted": results_data_distorted}.items():
            results_per_model["accuracies"].append( measures[model_key]["accuracy"] )
            results_per_model["jaccards"]  .append( measures[model_key]["jaccard"] )
            results_per_model["precisions"].append( measures[model_key]["precision"] )
            results_per_model["recalls"]   .append( measures[model_key]["recall"] )
            results_per_model["number_cells_predictions"].append( measures[model_key]["number_cells_predictions"] )        
        
    return parameter, results_original, results_data_augmented, results_data_distorted, number_cells_masks