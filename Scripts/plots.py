# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from utils import get_binary_predictions


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
    ax_mask.imshow(mask, 'gray')
    ax_mask.axis('off')
    plt.show()
    
    
def show_image_pred(image, model):
    """
    
    """
    pred = get_binary_predictions(image[None, :], model)
    show_image_mask(image, pred.squeeze())
    

def plot_all(results, parameter_name):
    """
    
    """
    parameter  = []
    accuracies = []
    jaccards   = []
    precisions = []
    recalls    = []
    number_cells_predictions = []
    number_cells_masks       = []
    
    for parameter_value, measures in results.items():
        parameter .append(parameter_value)
        accuracies.append(measures["accuracy"])
        jaccards  .append(measures["jaccard"])
        precisions.append(measures["precision"])
        recalls   .append(measures["recall"])
        number_cells_predictions.append(measures["number_cells_predictions"])
        number_cells_masks      .append(measures["number_cells_masks"])

    plt.figure()
    plt.title("Accuracy")
    plt.xlabel(parameter_name)
    plt.ylabel("Accuracy")
    plt.plot(parameter, accuracies)
    plt.grid()
    
    plt.figure()
    plt.title("Jaccard Score")
    plt.xlabel(parameter_name)
    plt.ylabel("Jaccard Score")
    plt.plot(parameter, jaccards)
    plt.grid()
    
    plt.figure()
    plt.title("Number of cells detected")
    plt.xlabel(parameter_name)
    plt.ylabel("Number of cells detected")
    plt.plot(parameter, number_cells_predictions)
    plt.plot(parameter, number_cells_masks, color='orange', linestyle='--')
    plt.legend(["Cells in Predictions", "Cells in Masks"])
    plt.grid()
    
    plt.figure()
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.plot(recalls, precisions)
    plt.scatter(recalls[0], precisions[0], color='orange', marker='o')
    plt.annotate(parameter[0], (recalls[0], precisions[0]), fontsize=12)
    plt.scatter(recalls[-1], precisions[-1], color='orange', marker='o')
    plt.annotate(parameter[-1], (recalls[-1], precisions[-1]), fontsize=12)
    plt.grid()