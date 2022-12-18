import math
import sklearn as sk
import numpy as np
import random
import sys
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

def get_percent(data, target, percent):
    size = int(data.shape[0] * float(percent) / 100.0)
    return (data[0:size],target[0:size])

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

def logit_func(W_mat, feature, k):
    ak = [0.0] * W_mat.shape[0]
    for i in range(0, W_mat.shape[0]):
        ak[i] = np.dot(W_mat[i], feature)
    denom = 0
    for i in range(0, W_mat.shape[0]):
        denom += safe_exp(ak[i])
    return float(safe_exp(ak[k])) / float(denom)

def cross_entropy_error(W_matrix, data, target):
    ans = 0
    for i in range(0, target.shape[0]):
        for j in range(0, W_matrix.shape[0]):
            if(target[i,j]==1):
                ans += logit_func(W_matrix, data[j],i)
    return -ans

def normalize_data(data):
    new_data = np.zeros(data.shape)
    for i in range(data.shape[0]):
        new_data[i] = normalize(data[i])
    return new_data

def safe_exp(x):
    if(x> 50):
        return 1
    if(x<-50):
        return -1
    return math.exp(x)

def normalize(vec):
    return vec / (((vec ** 2).sum()) ** .5)

def cross_entropy_error_gradient(W_matrix, data, target, k):
    ans = np.zeros(data.shape[1])
    for i in range(0, target.shape[0]):
        ans = ans + data[i] * (logit_func(W_matrix, data[i], k) - target[i,k])
    return ans

def gradient_descent(W_matrix, data, target, k, step_num, step_size):
    new_mat = np.zeros(W_matrix.shape)
    for i in range(0, W_matrix.shape[0]):
        new_mat[i] = W_matrix[i]
    for i in range(step_num):
        grad = normalize(cross_entropy_error_gradient(new_mat, data, target, k)) * step_size
        new_mat[k] = new_mat[k] - grad
    return new_mat

def calculate_probs(W_matrix, x, classes):
    probabilities = [0.0] * len(classes)
    for i in range(len(classes)):
        probabilities[i] = logit_func(W_matrix, x, i)
    return probabilities

def get_highest_prob(probs):
    highest = -1
    index = -1
    for i in range(len(probs)):
        if(probs[i] >= highest):
            index = i
            highest = probs[i]
    return index


def one_to_k(target, classes):
    new_target = np.zeros([target.shape[0], len(classes)])
    for i in range(target.shape[0]):
        new_target[i, int(target[i])] = 1
    return new_target

def lr_general(data_raw, target, classes, percents, times):
    hits = np.zeros([times, len(percents)])
    data = normalize_data(data_raw)
    for iter in range(times):
        (X_train_full, t_train_full, X_val, t_val) = get_80_20_split(data, target)
        for perc in range(len(percents)):
            (X_train, t_train_old) = get_percent(X_train_full, t_train_full, percents[perc])
            t_train = one_to_k(t_train_old, classes)

            W_matrix = np.random.rand(len(classes), data.shape[1])
            for i in range(len(classes)):
                W_matrix = gradient_descent(W_matrix, X_train, t_train, i, 100, .1)
            correct = 0
            for t in range(X_val.shape[0]):
                classification = get_highest_prob(calculate_probs(W_matrix, X_val[t], classes))
                if(classification == t_val[t]):
                    correct += 1
            hits[iter, perc] = float(correct)/float(X_val.shape[0])

            print("Run " + str(iter+1) + ", " + str(percents[perc]) + "% : " + str(1 - float(correct)/float(X_val.shape[0])))

    return hits

def logisticRegression(num_splits, percents):
    print("LOGISTIC REGRESSION")
    print("-------------------")
    print("Boston 50 Dataset")
    lr_general(X, get_50_target(t), [0,1], percents, num_splits)

    print("-----------------")
    print("Boston 75 Dataset")
    lr_general(X, get_75_target(t), [0,1], percents, num_splits)


    print("--------------")
    print("Digits Dataset")
    lr_general(X2, t2, [0,1,2,3,4,5,6,7,8,9], percents, num_splits)


logisticRegression(int(sys.argv[1]),  ast.literal_eval(sys.argv[2]))
