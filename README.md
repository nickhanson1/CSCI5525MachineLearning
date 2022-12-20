# CSCI 5525 Machine Learning Assignments

This repository is a collection of all of my assignments from the course CSCI 5525: Machine Learning, taught by Nicholas Johnson in the fall of 2020. 
All homeworks were written in Python. Each folder is one homework assignment, but contains multiple runnable files as well as a README
describing how to run each file. The files are as follows:

### HW1/logisticRegression.py
This creates a logistic regression model on the Boston Housing and MNIST datasets. Cross-validation has been implemented manually.

### HW1/naiveBayesGaussian.py
Creates a generative Gaussian model on the Boston Housing dataset.

### HW1/LDA1dThres.py
Uses Fischer's Linear Discriminant Analysis to reduce the Boston Housing dataset to one dimension, then finds a threshold to classify the data between above and below the median price.

### HW1/LDA2dGaussGM.py
Uses Fischer's Linear Discriminant Analysis to reduce the MNIST digits dataset down to a lower dimension, then creates a generative Gaussian model to classify the digits.

### HW2/SVM_dual.py
Implements a support vector machine for a two-class classification problem. This specific implementation solves a dual linear programming problem with slack variables.

### HW2/kernel_SVM.py
Implements a support vector machine, but applies a radial kernel before creating the support vector machine. 

### HW2/multi_SVM.py
Implements a support vector machine to classify the MNIST digits dataset.


### HW3/neural_net.py
Implements a neural network to classify the MNIST digits dataset. Uses cross entropy as a loss function and has one hidden layer of size 128.

### HW3/cnn.py
Implements a convolutional neural network. Has one convolutional layer for the first layer, a dropout layer, a fully connected layer, another dropout layer, and then a final fully connected layer. As for the previous network, it classifies the digits dataset.

### HW4/adaboost
Implements AdaBoost using stumps as weak learners. Classifies over a dataset of cancer patients, found here:
https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28original%29

### HW4/rf.py
Creates a random forest model with 100 decision stumps to classify over the cancer dataset.

### HW4/kmeans.py
Uses kmeans clustering to cluster together pixels with similar color values. Afterwards the image has each category of pixels be replaced with the mean color within that category, thus compressing the image.
