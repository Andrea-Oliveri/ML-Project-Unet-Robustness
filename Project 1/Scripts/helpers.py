# -*- coding: utf-8 -*-
"""Some helper functions."""
import csv
import numpy as np

# LOAD AND CREATE CSV:
# ------------------------------------------------------------------------------------------------------------------


def load_csv_data(data_path):
    """Loads data and returns yb (class labels), input_data (features) and ids (event ids).
    
    Args:
        data_path::[str]
            The file path of the CSV data file.
    Returns:
        yb::[np.array]
            The class labels.
        input_data::[np.array]
            The features.
        ids:[np.array]
            The ids.
    """
    y = np.genfromtxt(data_path,
                      delimiter=",",
                      skip_header=1,
                      dtype=str,
                      usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == 'b')] = -1

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """Creates an output file in csv format for submission to kaggle.
    
    Args:
        ids::[np.array]
            Event ids associated with each prediction.
        y_pred::[np.array]
            Predicted labels.
        name::[string]
            Filepath of the CSV output file to be created.
    """
    with open(name, 'w') as csvfile:
        fieldnames = ['Id', 'Prediction']
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({'Id': int(r1), 'Prediction': int(r2)})

            
# ------------------------------------------------------------------------------------------------------------------

# COMPUTE ACCURACY OF A MODEL & PREDICT LABELS FOR NEW INPUTS:
# ------------------------------------------------------------------------------------------------------------------


def compute_accuracy(predictions, real_labels):
    """Computes the accuracy between the predicted and the real labels.
    
    Args:
        predictions::[np.array]
            The values of the predicted labels for some input values.
        real_labels::[np.array]
            The real labels for these same input values.
    Returns:
        accuracy::[float]
            The percentage of good classification between the predictions and the real labels.
    """
    return np.mean(predictions == real_labels)


def predict_labels(w, tX):
    """Compute the predicted labels for regression methods.
    
    Args:
        w::[np.array]
            The weights of the current model.
        tX::[np.array]
            The unseen matrix of inputs.
    Returns:
        pred::[float]
            The label predictions (values <= 0 are predicted to be -1 and values > 0 are predicted to be 1).
    """
    pred = np.dot(tX, w)
    pred[pred <= 0] = -1
    pred[pred > 0] = 1

    return pred


def predict_labels_logit(w, tX, labels_0_1=False):
    """Compute the predicted labels for logistic regression.
    
    Args:
        w::[np.array]
            The weights of the current model.
        tX::[np.array]
            The unseen matrix of inputs.
        labels_0_1::[bool]
            The type of binary labels that we want. If True then the binary labels to output are {0,1} and if False
            the binary labels to output are {-1,1}. By default we ask for {-1,1} labels to output.
    Returns:
        pred::[float]
            The label predictions (if the predicted value (the probability of value y to be 1) is smaller than 0.5 
            then the predicted label will be 0 (or -1 if the second class is -1), and if the predicted value is greater 
            than 0.5 then the predicted label will be 1).
    """
    pred = 1 / (1 + np.exp(-tX.dot(w)))
    if labels_0_1:
        pred[pred <= 0.5] = 0
    else:
        pred[pred <= 0.5] = -1
    pred[pred > 0.5] = 1

    return pred


# ------------------------------------------------------------------------------------------------------------------