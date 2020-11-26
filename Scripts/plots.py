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
    

def plot_all(x, accuracies, jaccard_scores, number_cells_predictions, number_cells_masks):
    """
    
    """
    plt.figure()
    plt.title("Accuracy")
    plt.plot(x, accuracies)
    plt.grid()
    
    plt.figure()
    plt.title("Jaccard Score")
    plt.plot(x, jaccard_scores)
    plt.grid()
    
    plt.figure()
    plt.title("Number of cells detected")
    plt.plot(x, number_cells_predictions)
    if isinstance(number_cells_masks, int):
        plt.hlines(number_cells_masks, min(x), max(x), color='orange', linestyle='--')
    else:    
        plt.plot(x, number_cells_masks, color='orange', linestyle='--')
    plt.grid()