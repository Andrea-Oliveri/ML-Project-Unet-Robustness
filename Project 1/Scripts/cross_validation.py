# -*- coding: utf-8 -*-
"""Functions performing cross-validation for the different machine learning algorithms in implementations.py"""
import numpy as np
import implementations
from helpers import compute_accuracy, predict_labels, predict_labels_logit

# PARTITIONNING OF DATA FOR K-FOLD CROSS-VALIDATION
# ------------------------------------------------------------------------------------------------------------------


def partition_data(y, tX, n_folds):
    """Partitions output measures y and input measures tX, each into n_folds arrays.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        n_folds::[int]
            The number of partitions for each array.
    Returns:
        y_partitioned::[np.array]
            An array comprised of n_folds equally distributed arrays of y.
        tX_partitioned::[np.array]
            An array comprised of n_folds equally distributed arrays of tX.
    """
    random_order = np.random.permutation(y.size)
    random_tX = tX[random_order]
    random_y = y[random_order]
    return np.array_split(random_y, n_folds), np.array_split(random_tX, n_folds)


def get_train_validation_folds(y_part, tX_part, val_fold, n_folds):
    """Creates the training and validaton sets for k-fold Cross Validation.
    
    Args:
        y_part::[np.array]
            The output measures partitioned into n_folds.
        tX_part::[np.array]
            The input measures partitioned into n_folds.
        val_fold::[int]
            Index of the validation set, all other indexes are in the training set.
        n_folds::[int]
            The number of partitions.
    Returns:
        y_val::[np.array]
            Output measures in the validation set.
        tX_val::[np.array]
            Input measures in the validation set.
        y_train::[np.array]
            Output measures in the training set.
        tX_train::[np.array]
            Input measures in the training set.
    """
    y_val = y_part[val_fold]
    tX_val = tX_part[val_fold]
    y_train = np.concatenate(
        [y_part[fold] for fold in range(n_folds) if fold != val_fold])
    tX_train = np.concatenate(
        [tX_part[fold] for fold in range(n_folds) if fold != val_fold])
    return y_val, tX_val, y_train, tX_train


# ------------------------------------------------------------------------------------------------------------------

# 5-FOLD CROSS-VALIDATION FOR LEAST SQUARES METHODS TO TUNE HYPERPARAMETERS AND COMPUTE VALIDATION ACCURACY:
# ------------------------------------------------------------------------------------------------------------------


def least_squares_GD(y, tX, gamma_range, max_iters):
    """Performs cross-validation on Gradient Descent algorithm for least squares to obtain good step-size gamma as 
       well as an estimation of the validation accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        gamma_range::[(float, float)]
            A tuple containing the interval of step-size gamma we want to test.
        max_iters::[int]
            The maximum number of iterations allowed for the Gradient Descent algorithm.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation, its associated 
            training accuracy, its associated validation accuracy and its associated weight.
    """
    # Starting parameters
    w = np.zeros(tX.shape[1])
    gamma_start, gamma_end = gamma_range
    n_folds = 5
    n_gamma_trials = 9
    
    best_gamma = gamma_start
    best_loss = np.inf
    val_accuracy = 0

    # Partitioning of the data
    y_part, tX_part = partition_data(y, tX, n_folds)
    possible_gamma_values = np.geomspace(gamma_start, gamma_end,
                                         n_gamma_trials)

    for gamma in possible_gamma_values:
        with np.errstate(all='raise'):
            try:
                loss = 0
                accuracy = 0
                for val_fold in range(n_folds):
                    y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                        y_part, tX_part, val_fold, n_folds)

                    w_train, _ = implementations.least_squares_GD(
                        y_train, tX_train, w, max_iters, gamma)

                    loss += implementations.compute_loss_mse(
                        y_val, tX_val, w_train)
                    accuracy += compute_accuracy(
                        predict_labels(w_train, tX_val), y_val)

                if loss < best_loss:
                    best_gamma = gamma
                    best_loss = loss
                    val_accuracy = accuracy / n_folds

            except FloatingPointError:
                pass

    w, _ = implementations.least_squares_GD(y, tX, w, max_iters, best_gamma)
    train_accuracy = compute_accuracy(predict_labels(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "gamma": best_gamma
    }


def least_squares_SGD(y, tX, gamma_range, max_iters):
    """Performs cross-validation on Stochastic Gradient Descent algorithm for least squares to obtain good step-size 
       gamma as well as an estimation of the validation accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        gamma_range::[(float, float)]
            A tuple containing the interval of step-size gamma we want to test.
        max_iters::[int]
            The maximum number of iterations allowed for the Stochastic Gradient Descent algorithm.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation, its associated 
            training accuracy, its associated validation accuracy and its associated weight.
    """
    # Starting parameters
    w = np.zeros(tX.shape[1])
    gamma_start, gamma_end = gamma_range
    n_folds = 5
    n_gamma_trials = 9
    
    best_gamma = gamma_start
    best_loss = np.inf
    val_accuracy = 0

    # Partitioning of the data
    y_part, tX_part = partition_data(y, tX, n_folds)
    possible_gamma_values = np.geomspace(gamma_start, gamma_end,
                                         n_gamma_trials)

    for gamma in possible_gamma_values:
        with np.errstate(all='raise'):
            try:
                loss = 0
                accuracy = 0
                for val_fold in range(n_folds):
                    y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                        y_part, tX_part, val_fold, n_folds)

                    w_train, _ = implementations.least_squares_SGD(
                        y_train, tX_train, w, max_iters, gamma)

                    loss += implementations.compute_loss_mse(
                        y_val, tX_val, w_train)
                    accuracy += compute_accuracy(
                        predict_labels(w_train, tX_val), y_val)

                if loss < best_loss:
                    best_gamma = gamma
                    best_loss = loss
                    val_accuracy = accuracy / n_folds

            except FloatingPointError:
                pass

    w, _ = implementations.least_squares_SGD(y, tX, w, max_iters, best_gamma)
    train_accuracy = compute_accuracy(predict_labels(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "gamma": best_gamma
    }


def least_squares(y, tX):
    """Performs cross-validation on least_squares with normal equations to obtain an estimate of the validation accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
    Returns:
        best_values::[dict]
            A dictionary that contains the weights found with the normal equations, the associated training accuracy 
            and the associated estimated validation accuracy.
    """
    # Starting parameters
    n_folds = 5
    
    accuracy = 0

    # Partitioning of the data
    y_part, tX_part = partition_data(y, tX, n_folds)

    for val_fold in range(n_folds):
        y_val, tX_val, y_train, tX_train = get_train_validation_folds(
            y_part, tX_part, val_fold, n_folds)

        w_train, _ = implementations.least_squares(y_train, tX_train)

        accuracy += compute_accuracy(predict_labels(w_train, tX_val), y_val)

    val_accuracy = accuracy / n_folds

    w, _ = implementations.least_squares(y, tX)
    train_accuracy = compute_accuracy(predict_labels(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy
    }


# ------------------------------------------------------------------------------------------------------------------

# 5-FOLD CROSS-VALIDATION ON RIDGE REGRESSION TO TUNE HYPERPARAMETERS AND COMPUTE VALIDATION ACCURACY:
# ------------------------------------------------------------------------------------------------------------------


def ridge_regression(y, tX, lambda_range):
    """Performs cross-validation on ridge regression with normal equations to obtain a good weight of the regularisation
       term lambda as well as an estimation of the validation accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        lambda_range::[(float, float)]
            A tuple containing the interval of the weight of the regularisation term lambda we want to test.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation, the associated 
            weights found with the normal equations, the associated training accuracy and the associated estimated 
            validation accuracy.
    """
    # Starting parameters
    lambda_start, lambda_end = lambda_range
    n_folds = 5
    n_lambda_trials = 9
    
    best_lambda = lambda_start
    best_loss = np.inf
    val_accuracy = 0
    
    # Partitioning of the data
    y_part, tX_part = partition_data(y, tX, n_folds)
    possible_lambda_values = np.geomspace(lambda_start, lambda_end,
                                          n_lambda_trials)

    for lambda_ in possible_lambda_values:
        loss = 0
        accuracy = 0
        for val_fold in range(n_folds):
            y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                y_part, tX_part, val_fold, n_folds)

            w_train, _ = implementations.ridge_regression(
                y_train, tX_train, lambda_)

            loss += implementations.compute_loss_mse(y_val, tX_val, w_train)
            accuracy += compute_accuracy(predict_labels(w_train, tX_val),
                                         y_val)

        if loss < best_loss:
            best_lambda = lambda_
            best_loss = loss
            val_accuracy = accuracy / n_folds

    w, _ = implementations.ridge_regression(y, tX, best_lambda)
    train_accuracy = compute_accuracy(predict_labels(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "lambda": best_lambda
    }


# ------------------------------------------------------------------------------------------------------------------

# 5-FOLD CROSS-VALIDATION ON LOGISTIC REGRESSION TO TUNE HYPERPARAMETERS AND COMPUTE VALIDATION ACCURACY:
# ------------------------------------------------------------------------------------------------------------------


def logistic_regression(y, tX, gamma_range, max_iters):
    """Performs cross-validation on Stochastic Gradient Descent algorithm for logistic regression to obtain good step-size 
       gamma as well as an estimation of the validation accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        gamma_range::[(float, float)]
            A tuple containing the interval of step-size gamma we want to test.
        max_iters::[int]
            The maximum number of iterations allowed for the Stochastic Gradient Descent algorithm.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation, its associated 
            weights, its associated training accuracy and its associated estimated validation accuracy.
    """
    # Starting parameters
    w = np.zeros(tX.shape[1])
    gamma_start, gamma_end = gamma_range
    n_folds = 5
    n_gamma_trials = 9
    
    best_gamma = gamma_start
    best_loss = np.inf
    val_accuracy = 0

    # Convert labels to {0,1}
    y_logit = y.copy()
    y_logit[y_logit <= 0] = 0
    y_logit[y_logit > 0] = 1

    # Partitioning of the data
    y_part, tX_part = partition_data(y_logit, tX, n_folds)
    possible_gamma_values = np.geomspace(gamma_start, gamma_end,
                                         n_gamma_trials)

    for gamma in possible_gamma_values:
        with np.errstate(all='raise'):
            try:
                loss = 0
                accuracy = 0
                for val_fold in range(n_folds):
                    y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                        y_part, tX_part, val_fold, n_folds)

                    w_train, _ = implementations.logistic_regression(
                        y_train, tX_train, w, max_iters, gamma)

                    loss += implementations.compute_loss_logit(
                        y_val, tX_val, w_train)
                    accuracy += compute_accuracy(
                        predict_labels_logit(w_train, tX_val, labels_0_1=True),
                        y_val)

                if loss < best_loss:
                    best_gamma = gamma
                    best_loss = loss
                    val_accuracy = accuracy / n_folds

            except FloatingPointError:
                pass

    w, _ = implementations.logistic_regression(y_logit, tX, w, max_iters,
                                               best_gamma)
    train_accuracy = compute_accuracy(predict_labels_logit(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "gamma": best_gamma
    }


# ------------------------------------------------------------------------------------------------------------------

# 5-FOLD CROSS-VALIDATION ON REGULARISED LOGISTIC REGRESSION TO TUNE HYPERPARAMETERS AND COMPUTE VALIDATION ACCURACY:
# ------------------------------------------------------------------------------------------------------------------


def reg_logistic_regression(y, tX, gamma_range, lambda_range, max_iters):
    """Performs cross-validation on Stochastic Gradient Descent algorithm for reguralised logistic regression to 
       obtain good step-size gamma and a good weight for the regularisation term lambda as well as an estimation of 
       the validation accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        gamma_range::[(float, float)]
            A tuple containing the interval of step-size gamma we want to test.
        lambda_range::[(float, float)]
            A tuple containing the interval of the weight of the regularisation term lambda we want to test.
        max_iters::[int]
            The maximum number of iterations allowed for the Stochastic Gradient Descent algorithm.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation, its associated 
            weights, its associated training accuracy and its associated estimated validation accuracy.
    """
    # Starting parameters
    w = np.zeros(tX.shape[1])
    gamma_start, gamma_end = gamma_range
    lambda_start, lambda_end = lambda_range
    n_folds = 5
    n_gamma_trials = 9
    n_lambda_trials = 9
    
    best_lambda = lambda_start
    best_gamma = gamma_start
    best_loss = np.inf
    val_accuracy = 0

    # Convert labels to {0,1}
    y_logit = y.copy()
    y_logit[y_logit <= 0] = 0
    y_logit[y_logit > 0] = 1

    # Partitioning of the data
    y_part, tX_part = partition_data(y_logit, tX, n_folds)
    possible_gamma_values = np.geomspace(gamma_start, gamma_end,
                                         n_gamma_trials)
    possible_lambda_values = np.geomspace(lambda_start, lambda_end,
                                          n_lambda_trials)

    for lambda_ in possible_lambda_values:
        for gamma in possible_gamma_values:
            with np.errstate(all='raise'):
                try:
                    loss = 0
                    accuracy = 0
                    for val_fold in range(n_folds):
                        y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                            y_part, tX_part, val_fold, n_folds)

                        w_train, _ = implementations.reg_logistic_regression(
                            y_train, tX_train, lambda_, w, max_iters, gamma)

                        loss += implementations.compute_loss_logit(
                            y_val, tX_val, w_train, lambda_)
                        accuracy += compute_accuracy(
                            predict_labels_logit(w_train,
                                                 tX_val,
                                                 labels_0_1=True), y_val)

                    if loss < best_loss:
                        best_gamma = gamma
                        best_loss = loss
                        best_lambda = lambda_
                        val_accuracy = accuracy / n_folds

                except FloatingPointError:
                    pass

    w, _ = implementations.reg_logistic_regression(y_logit, tX, best_lambda, w,
                                                   max_iters, best_gamma)

    train_accuracy = compute_accuracy(predict_labels_logit(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "gamma": best_gamma,
        "lambda": best_lambda
    }


# ------------------------------------------------------------------------------------------------------------------

# 5-FOLD CROSS-VALIDATION ON K-NEAREST NEIGHBORS TO TUNE HYPERPARAMETERS AND COMPUTE VALIDATION ACCURACY:
# ------------------------------------------------------------------------------------------------------------------


def knn(y, tX, k_range):
    """Performs cross-validation on k-nearest neighbors to find a good k (number of neighbor(s) taken into account to 
       predict the label of the an unseen input) as well as obtaining an estimation of the validation accuracy.
       
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tX::[np.array]
            The input measures.
        k_range::[(float, float)]
            A tuple containing the interval of the number of neighbors to dertermine an unseen input k we want to test.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation and its associated
            estimated validation accuracy.
    """
    # Starting parameters
    k_start, k_end = k_range
    n_folds = 5
    n_k_trials = 3
    
    best_accuracy = 0
    best_k = k_start

    # Partitioning the data
    y_part, tX_part = partition_data(y, tX, n_folds)
    k_possible_values = np.unique(
        np.linspace(k_start, k_end, n_k_trials, dtype=int))

    for k in k_possible_values:
        accuracy = 0
        for val_fold in range(n_folds):
            y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                y_part, tX_part, val_fold, n_folds)

            predictions = implementations.knn(y_train, tX_train, tX_val, k)

            accuracy += compute_accuracy(predictions, y_val)

        accuracy /= n_folds

        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy

    return {"k": best_k, "val_accuracy": best_accuracy}


# ------------------------------------------------------------------------------------------------------------------

# 5-FOLD CROSS-VALIDATION ON SUPPORT VECTOR MACHINES TO TUNE HYPERPARAMETERS AND COMPUTE VALIDATION ACCURACY:
# ------------------------------------------------------------------------------------------------------------------


def svm(y, tX, gamma_range, lambda_range, max_iters):
    """Performs cross-validation on Stochastic Gradient Descent algorithm for support vector machines with soft margins to obtain good 
       step-size gamma and a good weight for the regularisation term lambda as well as an estimation of the validation 
       accuracy.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tX.
        tX::[np.array]
            The input measures.
        gamma_range::[(float, float)]
            A tuple containing the interval of step-size gamma we want to test.
        lambda_range::[(float, float)]
            A tuple containing the interval of the weight of the regularisation term lambda we want to test.
        max_iters::[int]
            The maximum number of iterations allowed for the Stochastic Gradient Descent algorithm.
    Returns:
        best_values::[dict]
            A dictionary that contains the best hyperparameter value found during the cross-validation, its associated 
            weights, its associated training accuracy and its associated estimated validation accuracy.
    """
    # Starting parameters
    w = np.zeros(tX.shape[1])
    gamma_start, gamma_end = gamma_range
    lambda_start, lambda_end = lambda_range
    n_folds = 5
    n_gamma_trials = 9
    n_lambda_trials = 9
    
    best_lambda = lambda_start
    best_gamma = gamma_start
    best_loss = np.inf
    val_accuracy = 0

    # Partitioning of the data
    y_part, tX_part = partition_data(y, tX, n_folds)
    possible_gamma_values = np.geomspace(gamma_start, gamma_end,
                                         n_gamma_trials)
    possible_lambda_values = np.geomspace(lambda_start, lambda_end,
                                          n_lambda_trials)

    for lambda_ in possible_lambda_values:
        for gamma in possible_gamma_values:
            with np.errstate(all='raise'):
                try:
                    loss = 0
                    accuracy = 0
                    for val_fold in range(n_folds):
                        y_val, tX_val, y_train, tX_train = get_train_validation_folds(
                            y_part, tX_part, val_fold, n_folds)

                        w_train, _ = implementations.svm(
                            y_train, tX_train, lambda_, w, max_iters, gamma)

                        loss += implementations.compute_loss_svm(
                            y_val, tX_val, w_train, lambda_)
                        accuracy += compute_accuracy(
                            predict_labels(w_train, tX_val), y_val)

                    if loss < best_loss:
                        best_gamma = gamma
                        best_lambda = lambda_
                        best_loss = loss
                        val_accuracy = accuracy / n_folds

                except FloatingPointError:
                    pass

    w, _ = implementations.svm(y, tX, best_lambda, w, max_iters, best_gamma)

    train_accuracy = compute_accuracy(predict_labels(w, tX), y)

    return {
        "w": w,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "gamma": best_gamma,
        "lambda": best_lambda
    }


# ------------------------------------------------------------------------------------------------------------------