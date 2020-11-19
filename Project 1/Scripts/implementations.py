# -*- coding: utf-8 -*-
"""Implementation of the tested methods."""
import numpy as np

# LEAST SQUARES:
# ------------------------------------------------------------------------------------------------------------------


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Applies least squares, with Gradient Descent algorithm.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        initial_w::[np.array]
            The starting value of the weights.
        max_iters::[int]
            The maximum number of iterations for Gradient Descent.
        gamma::[float]
            The step size value to use in the Gradient Descent.
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The MSE loss of the model associated to the best weights.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # calculate gradient
        grad = compute_gradient_least_squares(y, tx, w)
        # update w
        w = w - grad * (gamma / (1 + n_iter)**0.75)

    return w, compute_loss_mse(y, tx, w)


def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Applies least squares, with Stochastic Gradient Descent algorithm.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        initial_w::[np.array]
            The starting value of the weights.
        max_iters::[int]
            The maximum number of iterations for Stochastic Gradient Descent,
        gamma::[float]
            The step size value to use in the Stochastic Gradient Descent.
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The MSE loss of the model associated to the best weights.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # select a random measure to use to update
        random_index = np.random.choice(y.size)
        # calculate gradient
        grad = compute_gradient_least_squares(y[random_index], tx[random_index], w)
        # update w
        w = w - grad * (gamma / (1 + n_iter)**0.75)

    return w, compute_loss_mse(y, tx, w)


def least_squares(y, tx):
    """Applies least squares, using normal equations.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The MSE loss of the model associated to the best weights.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    return w, compute_loss_mse(y, tx, w)


def compute_gradient_least_squares(y, tx, w):
    """ Computes the gradient of the MSE loss function.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        w::[np.array]
            The current model's weights.       
    Returns:
        grad::[np.array]
            The gradient vector of the MSE loss.
    """
    e = y - tx @ w
    return -np.dot(e, tx) / y.size


def compute_loss_mse(y, tx, w):
    """Calculate the MSE loss.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        w::[np.array]
            The current model's weights.
    Returns:
        loss::[float]
            The MSE loss of the current model.
    """
    e = y - tx @ w
    return 0.5 * np.mean(e**2)


# ------------------------------------------------------------------------------------------------------------------

# RIDGE REGRESSION
# ------------------------------------------------------------------------------------------------------------------


def ridge_regression(y, tx, lambda_):
    """Applies ridge regression, with normal equations.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        lambda_::[float]
            The weight of the reguralisation term in the loss.
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The MSE loss of the model associated to the best weights.
    """
    w = np.linalg.solve(tx.T @ tx + 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1]), tx.T @ y)
    return w, compute_loss_mse(y, tx, w)


# ------------------------------------------------------------------------------------------------------------------

# LOGISTIC REGRESSION
# ------------------------------------------------------------------------------------------------------------------


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Applies logistic regression, with Stochastic Gradient Descent algorithm.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        initial_w::[np.array]
            The starting value of the weights.
        max_iters::[int]
            The maximum number of iterations for Stochastic Gradient Descent.
        gamma::[float]
            The step size value to use in the Stochastic Gradient Descent.
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The negative log likelihood loss of the model associated to the best weights.
    """
    return logistic_regression_SGD(y, tx, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Applies regularised logistic regression, with Stochastic Gradient Descent algorithm.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        lambda_::[float]
            The weight of the reguralisation term in the loss.
        initial_w::[np.array]
            The starting value of the weights.
        max_iters::[int]
            The maximum number of iterations for Stochastic Gradient Descent.
        gamma::[float]
            The step size value to use in the Stochastic Gradient Descent.       
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The negative log likelihood loss of the model associated to the best weights.
    """
    return logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, lambda_)


def logistic_regression_SGD(y, tx, initial_w, max_iters, gamma, lambda_=0):
    """Compute logistic regression with a regularisation term, Stochastic Gradient Descent algorithm.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        initial_w::[np.array]
            The starting value of the weights.
        max_iters::[int]
            The maximum number of iterations for Stochastic Gradient Descent.
        gamma::[float]
            The step size value to use in the Stochastic Gradient Descent.
        lambda_::[float]
            The weight of the reguralisation term in the loss. By default, the reguralisation term is 0 (no regularistion).     
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The negative log likelihood loss of the model associated to the best weights.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # select a random measure to use to update
        random_index = np.random.choice(y.size)
        # calculate gradient
        grad = compute_gradient_logit(y[random_index], tx[random_index], w, lambda_)
        # update w
        w = w - grad * (gamma / (1 + n_iter)**0.75)

    return w, compute_loss_logit(y, tx, w, lambda_)


def prediction_logit(tx, w):
    """Compute the probability for the output y to be 1 given a certain input tx and w. This function is used to predict 
       the value of the output given an unseen input given the weights found with a method.
       
    Args:
        tx::[np.array]
            The input measures.
        w::[np.array]
            The current weights of the model.
    Returns:
        predictions::[float]
            The probability for the output y to be 1.
    """
    return 1 / (1 + np.exp(-tx.dot(w)))


def compute_gradient_logit(y, tx, w, lambda_=0):
    """ Compute the gradient of the negative log likelihood loss function with a reguralisation term.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        w::[np.array]
            The current model's weights. 
        lambda::[float]
            The weight of the reguralisation term in the loss. By default, the reguralisation term is 0 (no regularistion).
    Returns:
        gradient::[np.array]
            The gradient vector of the negative log likelihood loss function with a reguralisation term.
    """
    return tx.T.dot(prediction_logit(tx, w) - y) + 2 * lambda_ * w


def compute_loss_logit(y, tx, w, lambda_=0):
    """ Compute the negative log likelihood loss function with a reguralisation term for logistic regression.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        w::[np.array]
            The current model's weights.
        lambda_::[float]
            The weight of the reguralisation term in the loss. By default, the reguralisation term is 0 (no regularistion).
    Returns:
        loss::[float]
            The negative log likelihood loss of the model associated to the best weights.
    """
    predictions = prediction_logit(tx, w)
    return -y.T.dot(np.log(predictions)) - (1 - y).T.dot(np.log(1 - predictions)) + lambda_ * w.T @ w


# ------------------------------------------------------------------------------------------------------------------


# K NEAREST NEIGHBOURS
# ------------------------------------------------------------------------------------------------------------------


def knn(y_train, x_train, x_new, k):
    """ Classifies each row of x using the k-nearest neighbours algorithm.
    
    Args:
        y_train::[np.array]
            The output measures associated to the input measures tx.
        x_train::[np.array]
            The input measures.
        x_new::[np.array]
            The new inputs that need to be classified.
        k::[int]
            The number of neighbours to use to infer the class of the new points.
    Returns:
        y_new::[np.array]
            The classes infered for the new inputs x_new.
    """
    if not k % 2:
        k -= 1
    if k <= 0:
        raise ValueError("Value of k must be >= 1")

    y = np.zeros(x_new.shape[0])
    for i in range(x_new.shape[0]):
        distances = np.linalg.norm(x_train - x_new[i], axis=1)
        neighbors = np.argpartition(distances, k)[:k]
        y[i] = np.median(y_train[neighbors])

    return y


# ------------------------------------------------------------------------------------------------------------------

# SVM
# ------------------------------------------------------------------------------------------------------------------


def svm(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Implementation of Support Vector Machines with soft margins using Stochastic Gradient Descent.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        tx::[np.array]
            The input measures.
        lambda_::[float]
            The weight of the reguralisation term in the loss.
        initial_w::[np.array]
            The starting value of the weights.
        max_iters::[int]
            The maximum number of iterations for Stochastic Gradient Descent.
        gamma::[float]
            The step size value to use in the Stochastic Gradient Descent.
    Returns:
        w::[np.array]
            The best weights found for the model.
        loss::[float]
            The Hinge loss with a regularisation term associated to the best weights.
    """
    w = initial_w
    for n_iter in range(max_iters):
        # select a random measure to use to update
        random_index = np.random.choice(y.size)
        # calculate gradient
        grad = compute_subgradient_svm(y[random_index], tx[random_index], w, lambda_, y.size)
        # update w
        w = w - grad * (gamma / (1 + n_iter)**0.75)
    return w, compute_loss_svm(y, tx, w, lambda_)


def compute_subgradient_svm(y, X, w, lambda_, num_samples):
    """ Compute the stochastic subgradient for Hinge loss with a reguralisation term.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        X::[np.array]
            The input measures.
        w::[np.array]
            The weights of the current model.
        lambda_::[float]
            The weight of the reguralisation term in the loss.
        num_examples::[int]
            The number of training points.
    Returns:
        subgradient::[np.array]
            A subgradient vector of the Hinge loss with a reguralisation term.
    """
    support = 1 - y * X @ w > 0
    return lambda_ * w - num_samples * support * X * y


def compute_loss_svm(y, X, w, lambda_):
    """ Compute the Hinge loss with a reguralisation term.
    
    Args:
        y::[np.array]
            The output measures associated to the input measures tx.
        X::[np.array]
            The input measures.
        w::[np.array]
            The weights of the current model.
        lambda_::[float]
            The weight of the reguralisation term in the loss.
    Returns:
        loss::[float]
            The Hinge loss with a reguralisation term for the model.
    """
    prod = y * (X @ w)
    return (lambda_ / 2) * (np.sum(w**2)) + np.sum(np.maximum(1 - prod, 0))


# ------------------------------------------------------------------------------------------------------------------