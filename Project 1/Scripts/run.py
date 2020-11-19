# -*- coding: utf-8 -*-
"""Script that runs the training and inference using the best method and
paramethers found."""
import numpy as np
from helpers import load_csv_data, predict_labels, create_csv_submission
from pretreatement import (impute_with_mean, remove_outliers, build_poly,
                           standardize, add_bias_column)
import cross_validation


# Setting seed for reproducibility.
np.random.seed(0)


# Definition of constants for the paths of useful files.
DATA_TRAIN_PATH = '../Dataset/train.csv'
DATA_TEST_PATH = '../Dataset/test.csv'
OUTPUT_PATH = '../Dataset/submission.csv'

# Definition of constants for the best parameters we found.
DEGREE = 5

# Loading training set.
print("Starting loading training set.")
y, X, _ = load_csv_data(DATA_TRAIN_PATH)
print("Training set loaded.\n")

# Pretreatement of training set.
print("Starting pretreatement of training set.")
X_new, imputed_mean_x = impute_with_mean(X)
X_new, y_new          = remove_outliers(X_new, y)
X_poly                = build_poly(X_new, DEGREE)
X_poly, mean_x, std_x = standardize(X_poly)
tX                    = add_bias_column(X_poly)
print("Performed pretreatement of training set.\n")


# Training model with least squares normal equation.
print("Starting training.")
results = cross_validation.least_squares(y_new, tX)
print("Training completed:\n",
      f"   Train accuracy     : {results['train_accuracy']:.5f}\n",
      f"   Validation accuracy: {results['val_accuracy']:.5f}\n")


# Loading test set.
print("Starting loading test set.")
_, X_test, ids_test = load_csv_data(DATA_TEST_PATH)
print("Test set loaded.\n")

# Pretreatement of test set.
print("Starting pretreatement of test set.")
X_test_new, _     = impute_with_mean(X_test, imputed_mean_x) 
X_test_poly       = build_poly(X_test_new, DEGREE)
X_test_poly, _, _ = standardize(X_test_poly, mean_x, std_x)
tX_test           = add_bias_column(X_test_poly)
print("Performed pretreatement of training set.\n")


# Infering labels of test set.
print("Starting inference.")
y_pred = predict_labels(results["w"], tX_test)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
print("Inference Completed.")

