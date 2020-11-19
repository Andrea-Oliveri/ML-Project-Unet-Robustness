# Higgs Boson Challenge - Machine Learning 2020 EPFL
Authors: Andrea Oliveri, Célina Chkroun, Bernardo Conde


## Introduction
This repository hosts our implementation of multiple Machine Learning algorithms in order to solve the Higgs Boson Challenge. This challenge consists of finding the best model to describe the actual CERN particle accelerator data to recreate the process of “discovering” the Higgs particle: From particle collision data, we had to estimate the likelihood that a given collision's "decay signature" (products that result from its decay process) was the result of the Higgs boson or of some other process/particle.


## Folder Structure
- `Dataset`: where all the CSV files for the training and the testing of the machine learning methods should be. 

- `Scripts`: contains all the Python3 code files. In this folder there are multiple important files:

    - `implementations.py`: contains the multiple algorithm tested in order to create various prediction models.

    - `cross_validation.py`: contains the k-fold cross validation for all the different methods implemented. The cross validation 
       algorithm is very useful to tune the different hyperparameters needed in the used methods and also to evaluate a valitdation 
       accuracy.

    - `pretreatment.py`: contains the different pretreatments tested.

    - `helpers.py`: have helper functions designed to work easier with the data (loading and creating CSV files), to predict the 
      label from the regression coefficient and a new input as well as computing of the accuracy of a model (% of correct predictions).

    - `plots.py`: contains functions for visualization to facilitate the analysis different informations on the features and hence to 
       help us better tackle the problem solving.

    - `higgs_boson.ipynb`: is the main core of the project, where each the pretreatments are applied in combination with each other or
       alone and after that the different methods are applied to these different pretreated dataset. The methods are called directly 
       through the cross-validation to return directly the regression coefficient with the best found hyperparameter. In this notebook, visualization is used to make feature analysis.

    - `run.py`: is a script that produces exactly the same predictions that we used in our best submission to AICrowd.
    

## Dependencies
Our project only depends on `python3`, `numpy`, `seaborn`, `matplotlib.pyplot` and `notebook`.


## Algorithms Implemented
- `implementations.py`:
	- Least Squares (with Gradient Descent, Stochastic Gradient Descent, and normal equations)
	  Linear regression of the data, using three different methods to find the regression coefficients. The Mean Squared Error has been 
	  used as the loss function.
	  
	- Ridge Regression (with normal equations)
	  Least Squares with L2-Regularization.

	- Logistic Regression (with Stochastic Gradient Descent)
	  Logistic regression of the data using Stochastic Gradient Descent to find the regression coefficients. The Negative Log Likelihood 
	  has been used as the loss function.

	- Regularized Logistic Regression (with Stochastic Gradient Descent)
	  Logistic regression of the data with L2-Regularization using Stochastic Gradient Descent to find the regression coefficients. The 
	  Negative Log Likelihood with the L2-Regularizer added has been used as the loss function.

	- K-Nearest Neighbour
	  Non-parametric method that classifies a new input with the predominant class of its K nearest neighbors.

	- Support Vector Machines with soft margins
	  Non-parametric method that finds a separating hyperplane that maximizes the margin between the two classes. There is a tolerance 
	  because the data are not separable. To find the regression coefficients, the Stochastic Subgradient descent has been used with the 
	  Hinge loss as the loss function.

	- Gradient Descent
	  Iterative optimization algorithm that modifies the regression coefficient in the opposite direction of the loss function's gradient 
	  over all samples in order to find a local minimum of a convex differentiable function.

	- Stochastic Gradient Descent
	  Iterative optimization algorithm that modifies the regression coefficient in the opposite direction of the loss function's gradient 
	  over one randomly chosen sample in order to find a local minimum of a convex differentiable function.

	- Stochastic Subgradient Descent
	  Iterative optimization algorithm that modifies the regression coefficient in the opposite direction of the loss function's subgradient 
	  over one randomly chosen sample in order to find a local minimum of a convex non-differentiable function.


- `cross_validation.py`:
	- K-fold Cross-Validation
	   K-fold cross-validation applied to tune the hyperparameters and estimate the validation accuracy. It makes K random folds of the 
	   data, and at each of the K iteration, K-1 folds are used as a training set and the left one as a test set.

