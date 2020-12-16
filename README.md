# Evaluation of Robustness of U-Net based Models to common Image Artefacts in Segmentation of Microscope Images - Machine Learning 2020
Authors: Andrea Oliveri, Célina Chkroun, Bernardo Conde

# Introduction
This repository hosts our canonical implementation and training of a Neural Network called [`U-Net`](https://arxiv.org/abs/1505.04597).
`U-Net` is a neural network with the special purpose of segmentating various neuronal structures in electron microscopic stacks.
We trained the net using three types of data generation from our original dataset and performed various types of analysis on the resulting
models to their robustness against multiple types of image artifacts.

# Folder Structure
- `Dataset`: directory storing the dataset.

- `Models`: directory containing three subdirectories in which the trained models are saved in the `ProtoBuffer` format.

- `Scripts`: contains all the `Python3` code files. In this folder there are multiple important files:
    - `UNET Train.ipynb` is a `Jupyter` notebook the training of all the `U-Net` models is performed.
  
    - `UNET Distortions Evaluate.ipynb` is a `Jupyter` notebook where multiple evaluations of the trained networks are done,
       mostly recording how well the network performs with artificially altered images.
  
    - `distortions.py` contains all the methods used to alter the testing images.

    - `image_processing_metods.py` contains class definitions performing segmentation via classical image processing techniques, namely
      applying a filter to remove the distortions and performing otsu thresholding.
  
    - `unet.py` is where our canonical implementation of the `U-Net` resides.
  
    - `utils.py` contains simple helper functions used by the rest of the project.

    - `plots.py` contains helper functions used to show images and plot results measured in the Jupyter notebooks.

    - `train_data_augmentation.py` contains helper functions applying different types of distorition to training images and returning
       resulting images as a generator to be called during models' training. 

# Dependencies
Our project depends on `python3`, `numpy`, `matplotlib.pyplot`, `notebook`, `opencv-python`, `tenserflow >= 2.1.0`

# Tensting environment
A .yml file was included in this repository describing the environment used for the training and testing. All tests were run on a
Windows 10 machine with an Intel Core i7-7700HQ CPU and Nvidia GeForce GTX 1050 GPU.