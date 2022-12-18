import math
import numpy as np
from sklearn.model_selection import train_test_split
import sys

###############################
#        DECISION STUMP       #
###############################
class Classifier:

    def __init__(self, X, t, weights, feature, classify):
        self.feature = feature
        self.classify = classify
        negative_weight = [0.0] * 10
        positive_weight = [0.0] * 10
        self.positive = []
        self.negative = []
        for i in range(X.shape[0]):
            if(t[i] == classify):
                positive_weight[int(X[i,feature]) - 1] += weights[i]
            else:
                negative_weight[int(X[i,feature]) - 1] += weights[i]
        for i in range(10):
            if(positive_weight[i]>negative_weight[i]):
                self.positive.append(i)
            else:
                self.negative.append(i)


    def stump_classify(self, data):
        #print(data[self.feature])
        #print(self.positive)
        if(data[self.feature]-1 in self.positive):
            return 1
        else:
            return -1




###############################
#      DATASET FUNCTIONS      #
###############################
def get_dataset(file):
    DB = np.genfromtxt(file,delimiter=',')
    X=DB[:,0:-1]
    t=DB[:,-1]
    return (X,t)

def remove_unlabelled_data(X,t):
    Xnew = X
    tnew = t
    for i in range(X.shape[0]-1, 0, -1):
        for j in range(X.shape[1]):
            if(math.isnan(Xnew[i,j]) or math.isnan(tnew[i])):
                tnew = np.delete(tnew, i, 0)
                Xnew = np.delete(Xnew, i, 0)
    return (Xnew, tnew)

def split_test_data(X, t, newdatasize):
    return (X[0:newdatasize-1],t[0:newdatasize-1], X[newdatasize:], t[newdatasize:])


def split_test_data_random(X, t, prop):
    data = np.append(X, t[np.newaxis].T, axis=1)
    Dtrain, Dtest = train_test_split(data, train_size = prop)
    return (Dtrain[:, :-1], Dtrain[:, -1], Dtest[:, :-1], Dtest[:,-1])

def print_raw_data(li):
    for l in li:
        print(l)

##############################
#      UTILITY FUNCTIONS     #
##############################

def lin_log(x, base):
    if(x == 0):
        return 0
    return x * math.log(x,2)

def get_feature_classes(X, feat):
    count_dict = {}
    for i in range(X.shape[0]):
        #if(X[i, feat] not in count_dict.keys()):
            #print("", X[i,feat], "     ", i, "   ", feat)
        if(X[i, feat] not in count_dict.keys()):
            count_dict[X[i, feat]] = 1
        else:
            count_dict[X[i, feat]] += 1
    return count_dict

def get_weighted_feature_classes(X, feat, w):
    count_dict = {}
    for i in range(X.shape[0]):
        if(X[i, feat] not in count_dict.keys()):
            count_dict[X[i, feat]] = w[i]
        else:
            count_dict[X[i, feat]] += w[i]
    return count_dict


def get_conditional_target_classes(X, t, feat, val, w):
    feat_classes = get_feature_classes(X, feat)
    c1 = 0
    c2 = 0
    for i in range(X.shape[0]):
        if(X[i, feat] == val):
            if(t[i] == 2):
                c1 += w[i]
            else:
                c2 += w[i]
    return (c1, c2)


def sign_classify(val, c):
    if(val == c):
        return 1
    else:
        return -1

##############################
# WEIGHTED ENTROPY FUNCTIONS #
##############################
def total_entropy(X, t, w):
    c1 = 0
    c2 = 0
    for i in range(t.shape[0]):
        if(t[i] == 2):
            c1 += w[i]
        if(t[i] == 4):
            c2 += w[i]
    return -(lin_log(c1,2) + lin_log(c2, 2))


def conditional_entropy_feature(X, t, feat, val, w):
    c1, c2 = get_conditional_target_classes(X, t, feat, val, w)

    return -(lin_log(c1,2) + lin_log(c2, 2))



def conditional_entropy(X, t, feat, w):
    feat_class = get_weighted_feature_classes(X, feat, w)
    entropy = 0
    for c in feat_class.keys():
        entropy += (feat_class[c] * conditional_entropy_feature(X, t, feat, c, w))
    return entropy

def information_gain(X, t, feat, w):
    return total_entropy(X, t, w) - conditional_entropy(X, t, feat, w)

##############################
#     ADABOOST FUNCTIONS     #
##############################

def ada_classify(x, alphas, stumps):
    classify = 0
    for i in range(len(alphas)):
        classify += alphas[i] * stumps[i].stump_classify(x)
    if(classify < 0):
        return -1
    else:
        return 1


def get_error(X, t, alphas, stumps, c):
    incorrect = 0
    for i in range(X.shape[0]):
        if(ada_classify(X[i], alphas, stumps) != sign_classify(t[i], c)):
            incorrect += 1
    return float(incorrect) / float(X.shape[0])


def get_weight_matrix(points, iterations):
    wmat = np.zeros((iterations, points))
    for i in range(points):
        wmat[0,i] = 1.0 / float(points)
    return wmat


def get_highest_feature_split(X, t, features, w):
    index = -1
    highest_ig = -1
    for i in range(features):
        ig = information_gain(X, t, i, w)
        if(highest_ig < ig):
            index = i
            highest_ig = ig
    return index


def calculate_classifier_error(X, t, stump, weights, c):
    error = 0
    for i in range(X.shape[0]):
        stumpc = stump.stump_classify(X[i])
        #print("", stumpc, "     ", sign_classify(t[i],c))
        if(stumpc != sign_classify(t[i],c)):
            error += weights[i]

    return .5 * math.log(float(1 - error) / error)


def perform_adaboost(X, t, Xtest, ttest, iterations):
    errors = []
    test_errors = []
    alphas = []
    stumps = []
    features = X.shape[1]
    weights = get_weight_matrix(X.shape[0], iterations + 1)
    classify = 2 # the value of target we will be testing for

    for i in range(iterations):
        feature_split = get_highest_feature_split(X, t, features, weights[i])               # split
        stump = Classifier(X, t, weights[i], feature_split, classify)                       # create the classifier
        alpha = calculate_classifier_error(X, t, stump, weights[i], classify)               # calculate the error
        norm = 0                                                                            #
        for j in range(X.shape[0]):                                                         #
            y = sign_classify(t[j], classify)                                               # Calculate the next set of weights
            weights[i + 1, j] = weights[i, j] * math.exp(-alpha * y * stump.stump_classify(X[j]))
            norm += weights[i + 1, j]                                                       #

        weights[i + 1] /= (norm + 1e-5)


        stumps.append(stump)
        alphas.append(alpha)
        errors.append(get_error(X, t, alphas, stumps, classify))
        test_errors.append(get_error(Xtest, ttest, alphas, stumps, classify))

    return (errors, test_errors)





def adaboost(file):
    #get dataset
    X, t = get_dataset(file)
    X, t = remove_unlabelled_data(X, t)
    #remove patient ids
    X = X[:,1:]
    #(X, t, Xtest, ttest) = split_test_data(X, t, 600)
    (X, t, Xtest, ttest) = split_test_data_random(X, t, .9)
    err, terr = perform_adaboost(X, t, Xtest, ttest, 100)
    print("Error rates:")
    for i in range(len(err)):
        print("    Iter ", i+1, ": ", err[i])
    print("Test Error rates: ")
    for i in range(len(terr)):
        print("    Iter ", i+1, ": ", terr[i])


if(len(sys.argv) < 2):
    print("Use: adaboost.py <filename>")
    exit(0)
adaboost(sys.argv[1])
