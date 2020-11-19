# -*- coding: utf-8 -*-
"""Functions plot some graphs to help feature analysis."""
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil


def plot_densities_all_features(X, y):
    """Plots the density of the labels for each features.
    
    Args:
        X::[np.array]
            The input measures.            
        y::[np.array]
            The output measures associated to the input measures X.
    """
    plt.rcParams.update({'font.size': 12})
    
    subplots_columns = 3
    subplots_number = X.shape[1]
    subplots_lines = ceil(subplots_number/subplots_columns)
    
    fig, axes = plt.subplots(subplots_lines, subplots_columns, figsize=(21, 50))
    axes = axes.flatten()
    plt.subplots_adjust(hspace=0.35)

    for i,feature in enumerate(X.T):
        plot_density(feature, y, axes[i], f"Feature {i+1}")
    plt.show()

    
def plot_correlation_feature_label(corr_coeffs, thr):
    """Plots the correlation coefficient of each features with the labels.
    
    Args:
        corr_coeffs::[np.array]
            Correlation coefficients of the features with the labels.
        thr::[float]
            The threshold value at which to draw the line on the graph.
    """
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(10, 5))
    
    # plot corr_coeffs for each feature
    features = range(1, len(corr_coeffs)+1)
    plt.bar(features, abs(corr_coeffs), tick_label=features)
    plt.ylabel("Pearson's Correlation Coefficient")
    plt.xlabel('Feature')
    plt.title("Correlation coefficient between features and labels")
    
    # add the threshold line
    plt.plot((0.5, 30.5),(thr, thr), "--", color="red")
    
    plt.show()
    

def plot_density(feature, y, ax, title):
    """Plots the density of the labels for one feature.
    
    Args:
        feature::[np.array]
            The feature with which we want to make the density plot.
        y::[np.array]
            The labels of the measures.
        ax::[matplotlib.axes]
            The axe we want to plot on.
        title::[str]
            The title we want to give to the plot.
    """
    classes = (feature[y < 0], feature[y > 0])
    
    for subset in classes:
        sns.kdeplot(subset, fill=True, linewidth=2, bw_adjust=0.2, ax=ax)

    ax.set_ylabel('Density')
    ax.set_xlabel('Values of the feature')
    ax.set_title(title)