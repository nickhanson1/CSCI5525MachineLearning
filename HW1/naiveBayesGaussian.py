import math
import sklearn as sk
import numpy as np
import random
import sys
import ast
import matplotlib.pyplot as plt

from sklearn import datasets
X, t = datasets.load_boston(return_X_y=True)
X2,t2 = datasets.load_digits(return_X_y=True)

#
#Create Boston50 dataset target
def get_50_target(target):
    median50 = (np.sort(target)[int(target.shape[0] * .5)])
    target50 = np.zeros(target.shape[0])
    for i in range(target50.shape[0]):
        if(target[i] >= median50):
            target50[i]=1

    sum = 0
    return target50

#
#Create Boston75 dataset target
def get_75_target(target):
    median75 = (np.sort(target)[(int)(target.size * .75)])
    target75 = np.zeros(target.size)
    for i in range(target75.size):
        if(target[i] >= median75):
            target75[i]=1

    return target75

def get_80_20_split(data, target):
    datasize = data.shape[0]
    features = data.shape[1]
    trainsize = int(datasize * .8)
    X_train = np.zeros([trainsize, features])
    t_train = np.zeros([trainsize])
    X_val = np.zeros([datasize - trainsize, features])
    t_val = np.zeros(datasize - trainsize)
    enum_train = list(range(0, datasize))
    enum_val = []
    for i in range(datasize - trainsize):
        choice = random.randint(0, len(enum_train) - 1)
        del enum_train[choice]
        enum_val.append(choice)
    enum_val.sort()
    random.shuffle(enum_train)
    for i in range(len(enum_train)):
        X_train[i] = data[enum_train[i]]

        t_train[i] = target[enum_train[i]]
    for i in range(len(enum_val)):
        X_val[i]=data[enum_val[i]]
        t_val[i]=target[enum_val[i]]

    return (X_train, t_train, X_val, t_val)

def get_percent(data, target, percent):
    size = int(data.shape[0] * float(percent) / 100.0)
    return (data[0:size],target[0:size])





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
        class_totals[int(target[i])] = class_totals[int(target[i])] + 1
    return class_totals




def safe_inverse(mat):
    if(np.linalg.det(mat)==0):
        make_invertible = np.identity(mat.shape[0]) * .005
        return np.linalg.inv(mat+make_invertible)
    else:
        return np.linalg.inv(mat)



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


def square_entries(vec):
    new_mat = np.zeros(vec.shape[0])
    for i in range(vec.shape[0]):
        new_mat[i] = vec[i] ** 2
    return new_mat

def calculate_feature_averages(data, target, class_totals):
    avg = np.zeros([data.shape[1], len(class_totals)])
    for i in range(data.shape[0]):
        cl = int(target[i])
        avg[:,cl] = avg[:,cl]+data[i]

    for i in range(len(class_totals)):
        avg[:,i] = avg[:,i] / float(class_totals[i])

    return avg


def normalize(vec):
    return vec / (((vec ** 2).sum()) ** .5)

def calculate_feature_variances(data, target, class_totals, averages):
    cov = np.zeros([data.shape[1], len(class_totals)])
    count = np.zeros([data.shape[1], len(class_totals)])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            c = int(target[i])
            cov[j,c] = cov[j,c] + (data[j,c]-averages[j,c]) ** 2
            count[j,c] = count[j,c]+1
    for j in range(data.shape[1]):
        for c in range(len(class_totals)):
            cov[j,c] = cov[j,c] / count[j,c]
    return cov



def calculate_logistic_input(x, averages, covariances, priors, k):
    N = averages.shape[0]
    answer = - N * (.5) * math.log(2 * math.pi)
    for i in range(N):
        if(covariances[i,k] != 0):
            answer = answer - math.log(math.sqrt(covariances[i, k]))
            answer = answer + (((x[i] - averages[i,k]) ** 2) / covariances[i, k])


    answer = answer + math.log(priors[k])
    return answer

def normalize_data(data):
    new_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        new_data[i] = normalize(data[i])
    return new_data

def safe_exp(x):
    if(x<-50):
        return -1
    return math.exp(x)

def compute_gauss(x, mean, variance):
    if(variance == 0):
        return 1
    exponent = -(.5) * ((x - mean) ** 2) / (variance)
    if(math.exp(exponent == 0)):
        return 1
    return (1 / math.sqrt(2 * math.pi * variance))  * math.exp(exponent)

def softmax(a_vals, k):
    return a_vals[k] / np.sum(a_vals)

def get_posterior_probabilities(x, averages, variances, priors):
        class_size = len(priors)
        posteriors = [0.0] * class_size
        a_vals = [0.0] * class_size
        feature_size = averages.shape[0]
        for k in range(class_size):
            for i in range(feature_size):
                normal = compute_gauss(x[i], averages[i,k], variances[i,k])
                a_vals[k] = a_vals[k] + math.log(normal)
            a_vals[k] = a_vals[k] + math.log(priors[k])
        for k in range(class_size):
            posteriors[k] = softmax(a_vals, k)
        return posteriors

def get_highest_prob(probs):
    highest = -1
    index = -1
    for i in range(len(probs)):
        if(probs[i] >= highest):
            index = i
            highest = probs[i]
    return index

def nbg_general(data_raw, target, classes, percents, times):
    hits = np.zeros([times, len(percents)])
    data = normalize_data(data_raw)
    for iters in range(times):
        (X_train_full, t_train_full, X_val, t_val) = get_80_20_split(data, target)
        for perc in range(len(percents)):
            (X_train, t_train) = get_percent(X_train_full, t_train_full, percents[perc])
            class_totals = calculate_class_totals(t_train, classes)
            class_priors = generative_calculate_priors(X_train.shape[0], class_totals)

            feature_averages_matrix = calculate_feature_averages(X_train, t_train, class_totals)
            feature_variance_matrix = calculate_feature_variances(X_train, t_train, class_totals, feature_averages_matrix)


            correct = 0
            for j in range(X_val.shape[0]):
                posteriors = get_posterior_probabilities(X_val[j], feature_averages_matrix, feature_variance_matrix, class_priors)
                classify = get_highest_prob(posteriors)
                if(int(t_val[j]) == classify):
                    correct = correct + 1
            print("Run " + str(iters+1) + ", " + str(percents[perc]) + "% : " + str(1 - float(correct)/float(X_val.shape[0])))
            hits[iters, perc] = float(correct)/float(X_val.shape[0])

    return hits


def logisticRegression(num_splits, percents):
    print("NAIVE BAYES WITH MARGINAL GAUSSIAN DIST")
    print("-------------------")
    print("Boston 50 Dataset")
    nbg_general(X, get_50_target(t), [0,1], percents, num_splits)

    print("-----------------")
    print("Boston 75 Dataset")
    nbg_general(X, get_75_target(t), [0,1], percents, num_splits)


    print("--------------")
    print("Digits Dataset")
    nbg_general(X2, t2, [0,1,2,3,4,5,6,7,8,9], percents, num_splits)

logisticRegression(int(sys.argv[1]), ast.literal_eval(sys.argv[2]))
