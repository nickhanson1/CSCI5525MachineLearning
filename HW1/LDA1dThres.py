import math
import sklearn as sk
import numpy as np
import sys
import matplotlib.pyplot as plt

from sklearn import datasets
X, t = datasets.load_boston(return_X_y=True)

#
#Create Boston50 dataset target
def get_50_target(target):
    median50 = (np.sort(target)[(int)(target.size * .5)])
    target50 = np.zeros(target.size)
    for i in range(target50.size):
        if(target[i] >= median50):
            target50[i]=1

    return target50



#
#  Returns 4-tuple with the dataset and target to train on and the dataset and target to validate on
#
def get_fold(i, folds, data, target):
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

def calculate_averages(data, target, datasize, featuresize):
    m1 = np.zeros(featuresize)
    class1 = 0.0
    m2 = np.zeros(featuresize)
    class2 = 0.0
    for i in range(datasize):
        if(target[i] == 0):
            class1 += 1.0
            m1 += data[i]
        else:
            class2 += 1.0
            m2 += data[i]
    m1 *= (1 / class1)
    m2 *= (1 / class2)
    return (m1,m2)

def calculate_between_covariance(avg1, avg2):
    difference = np.subtract(avg2, avg1)
    return np.dot(difference, np.transpose(difference))

def calculate_within_covariance(avg1, avg2, data, target, datasize, featuresize):
    S_W = np.zeros([featuresize,featuresize])
    for i in range(0, datasize):
        diff = None
        if(target[i] == 0):
            diff = np.subtract(data[i],avg1)[np.newaxis]
        if(target[i] == 1):
            diff = np.subtract(data[i],avg2)[np.newaxis]

        cov = np.dot(diff.T, diff)
        S_W = S_W + cov

    return S_W

#
# Returns the median of the data and true if most data whose targets are 0 are mapped to below
# the median and true otherwise
#
def find_threshold(data, target, projection, datasize):
    m = np.zeros(data.shape[1])
    for i in range(datasize):
        m += data[i]
    m = (1 / datasize) * m
    return np.dot(projection, m)


def norm(w):
    norm = 0
    for i in range(w.shape[0]):
        norm += w[i] ** 2
    return math.sqrt(norm)


def LDA1dThres(folds):
    target = get_50_target(t)
    errors = [0.0] * folds
    for i in range(0, folds):
        (X_train, t_train, X_val, t_val) = get_fold(i, folds, X, target)

        datasize = X_train.shape[0]
        featuresize = X_train.shape[1]
        (avg1, avg2) = calculate_averages(X_train, t_train, datasize, featuresize)
        naiveLDA = avg2 - avg1
        SW = calculate_within_covariance(avg1, avg2, X_train, target, datasize, featuresize)
        w = np.dot(np.linalg.inv(SW), naiveLDA)
        w0  = find_threshold(X_train, t_train, w, datasize)

        correctly_predicted = 0
        for j in range(0, t_val.shape[0]):
            prediction = np.dot(X_val[j],w)-w0
            if(prediction <= 0 and t_val[j] == 0):
                correctly_predicted = correctly_predicted + 1
            if(prediction > 0 and t_val[j] == 1):
                correctly_predicted = correctly_predicted + 1
        errors[i] = correctly_predicted / (t_val.shape[0])

    avg = 0
    for i in errors:
        avg += i
    avg /= folds
    var = 0
    for i in errors:
        var += (i - avg) ** 2
    var /= folds
    print("The test set error rates are:")
    for i in range(len(errors)):
        print(str(i+1) + ": " + str(1-errors[i]))
    print("The average error rate is " + str(1-avg))
    print("The standard deviation of the error rates is " + str(math.sqrt(var)))


LDA1dThres(int(sys.argv[1]))
