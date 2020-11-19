# -*- coding: utf-8 -*-
"""Functions performing different pretreatements of the dataset."""
import numpy as np


def drop_uncomplete_measures(X, y):
    """Removes all measures that contain missing features from the dataset.
    
    Args:
        X::[np.array]
            The input measures.
        y::[np.array]
            The output measures associated to the input measures X.

    Returns:
        good_X::[np.array]
            The filtered input measures.
        good_Y::[np.array]
            The filtered output measures.
    """
    missed_measure = X < -900
    good_x = np.logical_not(missed_measure.any(axis=1))
    return X[good_x], y[good_x]


def add_bias_column(X):
    """Adds a bias column to the supplied matrix.
    
    Args:
        X::[np.array]
            The input measures.
    Returns:
        bias_X::[np.array]
            The input measures with a bias column added.
    """
    return np.c_[np.ones(X.shape[0]), X]


def standardize(X, mean_x=None, std_x=None):
    """Standardize the original data set.
    
    Args:
        X::[np.array]
            The input measures.
        mean_x::[float]
            The mean of the input measures. By default, it is None, and if this is the case it will be computed.
        std_x::[float]
            The standard deviation of the input measures. By default, it is None, and if this is the case it will 
            be computed.
    Returns:
        standard_X::[np.array]
            The standardized input measures.
    """
    if mean_x is None:
        mean_x = X.mean(axis=0)
    if std_x is None:
        std_x = X.std(axis=0)

    return (X - mean_x) / std_x, mean_x, std_x


def remove_outliers(X, y):
    """ Remove all outliers from the data set.
    
    Args:
        X::[np.array]
            The input measures.
        y::[np.array]
            The output measures associated to the input measures X.
    Returns:
        good_X::[np.array]
            The input measures without outliers.
        good_Y::[np.array]
            The output measures without outliers.
    """
    mean_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    outlier = np.abs(X - mean_x) >= 3 * std_x
    good_x = np.logical_not(outlier.any(axis=1))
    return X[good_x], y[good_x]


def impute_with_mean(X, mean_x=None):
    """Returns a copy of the array passed as parameter, with all missed measures set to the mean of the feature.
    
    Args:
        X::[np.array]
            The input measures.
        mean_x::[float]
            The mean of the input measures. By default, it is None, and if this is the case it will be computed.
    Returns:
        X_new::[np.array]
            The input measures with missing measures set as the mean of the features.
    """
    X_new = X.copy()
    missing_features = np.where(X_new < -900)
    X_new[missing_features] = np.nan
    if mean_x is None:
        mean_x = np.nanmean(X_new, axis=0)
    X_new[missing_features] = np.take(mean_x, missing_features[1])
    return X_new, mean_x


def build_poly(X, degree):
    """Polynomial basis functions for input data x, for j=1 up to j=degree.
    
    Args:
        X::[np.array]
            The input measures.
        degree::[int]
            The greater degree of the polynoma.
    Returns:
        poly::[np.array]
            The augmented input measures.
    """
    poly = X
    for deg in range(2, degree + 1):
        poly = np.c_[poly, np.power(X, deg)]
    return poly
