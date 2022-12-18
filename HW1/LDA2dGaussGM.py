import sklearn as sk
import numpy as np
import copy
import math

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import sys

from sklearn import datasets
X, t = datasets.load_digits(return_X_y=True)



def safe_inverse(mat):
    if(np.linalg.det(mat)==0):
        make_invertible = np.identity(mat.shape[0]) * .000001
        return np.linalg.inv(mat+make_invertible)
    else:
        return np.linalg.inv(mat)

def normalize(vec):
    return vec / (((vec ** 2).sum()) ** .5)


def get_fold(data, target, i, folds):
    datasize = data.shape[0]
    foldsize = (int)((datasize) / folds)
    if(i < folds - 1):
        X_val = data[i * foldsize : ((i + 1) * foldsize)]
        t_val = target[i * foldsize : ((i + 1) * foldsize)]
        X_train = np.zeros([datasize - foldsize, data.shape[1]])
        t_train = np.zeros(datasize - foldsize)
        for j in range(0, i * foldsize):
            X_train[j]=data[j]
            t_train[j]=target[j]
        for j in range((i + 1) * foldsize, datasize):
            X_train[j-(foldsize)]=data[j]
            t_train[j-(foldsize)]=target[j]
        return (X_train, t_train, X_val, t_val)

    if(i == folds - 1):
        X_val = data[(folds - 1) * foldsize + 1 : ]
        t_val = target[(folds - 1) * foldsize + 1 : ]
        X_train = data[0 : (folds - 1) * foldsize]
        t_train = target[0 : (folds - 1) * foldsize]
        return (X_train, t_train, X_val, t_val)

def calculate_total_average(data):
    avg = np.zeros(data.shape[1])
    for i in range(data.shape[0]):
        avg += data[i]
    return avg / data.shape[0]

def calculate_class_averages(data, target, class_totals):
    avgs = [np.zeros(data.shape[1])] * len(class_totals)
    for i in range(data.shape[0]):
        c = int(target[i])
        avgs[c] = avgs[c] + data[i]

    for i in range(len(class_totals)):
        avgs[i] = avgs[i] * (1 / float(class_totals[i]))
    return avgs

def calculate_class_totals(target, classes):
    class_totals = [0] * len(classes)
    for i in range(target.shape[0]):
        class_totals[int(target[i])] += 1
    return class_totals

def calculate_within_covariance(data, target, class_averages):
    SW = np.zeros([data.shape[1], data.shape[1]])
    for i in range(data.shape[0]):
        c = int(target[i])
        diff = (data[i] - class_averages[c])[np.newaxis]
        cov = np.dot(diff.T, diff)
        SW = SW + cov
    return SW

def calculate_between_covariance(data, target, class_averages, class_totals, total_average):
    SB = np.zeros([data.shape[1], data.shape[1]])
    for i in range(len(class_totals)):
        diff = (class_averages[i] - total_average)[np.newaxis]
        cov = class_totals[i] * (np.dot(diff.T, diff))
        SB = SB + cov
    return SB

def get_largest_eigenvectors(vals, vecs):
    vals = np.real(vals)
    vecs = np.real(vecs)
    vec1 = None
    vec2 = None
    sorted_vals = np.sort(copy.deepcopy(vals))
    list_length = len(vals)
    for i in range(len(vals)):
        if(vals[i] == sorted_vals[list_length - 2]):
            vec1 = vecs[i]
        if(vals[i] == sorted_vals[list_length - 3]):
            vec2 = vecs[i]
    return (vec1, vec2)

def generative_calculate_covariances(datap, target, class_averages, class_totals):
    covariances = [np.zeros([datap.shape[1], datap.shape[1]])] * len(class_totals)
    for i in range(datap.shape[0]):
        cl = int(target[i])
        diff = (datap[i] - class_averages[cl])[np.newaxis]
        covariances[cl] = covariances[cl] + (np.dot(diff.T, diff))

    for k in range(len(class_totals)):
        covariances[k] = covariances[k] * (1 / class_totals[k])

    return covariances

def generative_calculate_priors(datasize, class_totals):
    priors = [0.0] * len(class_totals)
    for i in range(len(class_totals)):
        priors[i] = float(class_totals[i]) / float(datasize)
    return priors

def generative_construct_conditionals(priors, averages, covariances):
    w_list = [None] * len(averages)
    w0_list = [None] * len(averages)
    for i in range(len(priors)):
        precision = safe_inverse(covariances[i])
        w_list[i] = np.dot(precision, averages[i])
        w0_list[i] = -(.5) * np.dot(averages[i], np.dot(precision, averages[i])) + math.log(priors[i])
    return (w_list, w0_list)

def calculate_prob_from_weights(x, i, w_list, w0_list):
    numerator = math.exp(np.dot(w_list[i], x) + w0_list[i])
    denominator = 0
    for j in range(len(w_list)):
        denominator += math.exp(np.dot(w_list[j], x) + w0_list[j])
    return numerator / denominator

def get_prob_from_conditionals(x, w_list, w0_list):
    highest = -1
    index = -1
    for i in range(len(w_list)):
        prob = calculate_prob_from_weights(x, i, w_list, w0_list)
        if(prob >= highest):
            index = i
            highest = prob
    return index

def LDA2dGaussGM(folds):
    target = t
    classes = [0,1,2,3,4,5,6,7,8,9]
    errors = [0.0] * folds
    for fold_iter in range(0, folds):
        (X_train, t_train, X_val, t_val) = get_fold(X, target, fold_iter, folds)
        datasize = X_train.shape[0]
        featuresize = X_train.shape[1]
        total_average  = calculate_total_average(X_train)
        class_totals = calculate_class_totals(t_train, classes)
        class_averages = calculate_class_averages(X_train, t_train, class_totals)
        SW = calculate_within_covariance(X_train, t_train, class_averages)
        SB = calculate_between_covariance(X_train, t_train, class_averages, class_totals, total_average)

        plane_determine = np.dot(safe_inverse(SW), SB)

        (eigenvalues, eigenvectors) = np.linalg.eig(plane_determine)
        (vec1, vec2) = get_largest_eigenvectors(eigenvalues, eigenvectors)
        vec1 = normalize(vec1)
        vec2 = normalize(vec2)
        projection_matrix = np.zeros([2, vec1.shape[0]])
        projection_matrix[0] = vec1
        projection_matrix[1] = vec2




        proj_data = np.zeros([X_train.shape[0], 2])
        for j in range(proj_data.shape[0]):
            proj_data[j] = np.dot(projection_matrix, X_train[j])

        #code for displaying the projection

        colors = cm.rainbow(np.linspace(0, 1, 10))
        for j in range(proj_data.shape[0]):
            plt.scatter(proj_data[j,0], proj_data[j,1], color=colors[int(t_train[j])])
        plt.show()

        class_proj_averages = calculate_class_averages(proj_data, t_train, class_totals)
        class_covariances = generative_calculate_covariances(proj_data, t_train, \
                                                            class_proj_averages, class_totals)

        class_priors = generative_calculate_priors(proj_data.shape[0], class_totals)

        (weight_list, bias_list) = generative_construct_conditionals(class_priors, \
                                                        class_proj_averages, class_covariances)

        correct = 0
        for j in range(X_val.shape[0]):
            projected = np.dot(projection_matrix, X_val[j])
            choice = get_prob_from_conditionals(projected, weight_list, bias_list)
            #print(choice)
            if(choice == int(t_val[j])):
                correct += 1

        errors[fold_iter] = float(correct) / float(X_val.shape[0])

    total_avg = 0
    for i in range(len(errors)):
        total_avg += errors[i]
        print("Fold " + str(i+1) + " error rate : " + str(1-errors[i]))
    total_avg /= folds
    var = 0
    for i in range(len(errors)):
        var += (total_avg - errors[i]) ** 2
    var /= len(errors)
    print("Total Average Error : " + str(1-total_avg))
    print("Total Variance : " + str(var))


LDA2dGaussGM(int(sys.argv[1]))
