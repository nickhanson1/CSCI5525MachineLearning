import math
import numpy as np
import random
from sklearn.model_selection import train_test_split
import sys

###############################
#        DECISION STUMP       #
###############################
class Classifier:

    def __init__(self, X, t, feature, classify):
        self.feature = feature
        self.classify = classify
        negative_weight = [0.0] * 10
        positive_weight = [0.0] * 10
        self.positive = []
        self.negative = []
        for i in range(X.shape[0]):
            if(t[i] == classify):
                positive_weight[int(X[i,feature]) - 1] += 1
            else:
                negative_weight[int(X[i,feature]) - 1] += 1
        for i in range(10):
            if(positive_weight[i]>negative_weight[i]):
                self.positive.append(i)
            else:
                self.negative.append(i)


    def stump_classify(self, data):
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
        if(X[i, feat] not in count_dict.keys()):
            count_dict[X[i, feat]] = 1
        else:
            count_dict[X[i, feat]] += 1
    return count_dict


def get_conditional_target_classes(X, t, feat, val):
    feat_classes = get_feature_classes(X, feat)
    c1 = 0
    c2 = 0
    for i in range(X.shape[0]):
        if(X[i, feat] == val):
            if(t[i] == 2):
                c1 += 1
            else:
                c2 += 1
    return (c1, c2)


def sign_classify(val, c):
    if(val == c):
        return 1
    else:
        return -1


##############################
#     ENTROPY FUNCTIONS      #
##############################
def total_entropy(X, t):
    c1 = 0
    c2 = 0
    for i in range(t.shape[0]):
        if(t[i] == 2):
            c1 += 1
        if(t[i] == 4):
            c2 += 1
    total = c1 + c2
    p1 = float(c1) / float(total)
    p2 = float(c2) / float(total)
    return -(lin_log(p1, 2) + lin_log(p2, 2))


def conditional_entropy_feature(X, t, feat, val):
    c1, c2 = get_conditional_target_classes(X, t, feat, val)
    p1 = float(c1) / float(c1 + c2)
    p2 = float(c2) / float(c1 + c2)
    return -(lin_log(p1, 2) + lin_log(p2, 2))



def conditional_entropy(X, t, feature):
    feat_class = get_feature_classes(X, feature)
    total = 0
    for c in feat_class.keys():
        total += feat_class[c]
    entropy = 0
    for c in feat_class.keys():
        entropy += ((float(feat_class[c]) / float(total)) * conditional_entropy_feature(X, t, feature, c))
    return entropy

def information_gain(X, t, feat):
    return total_entropy(X, t) - conditional_entropy(X, t, feat)




###############################
#    RANDOM TREE FUNCTIONS    #
###############################

def get_error(X, t, ensemble, c):
    incorrect = 0
    for i in range(X.shape[0]):
        if(random_forest_classify(X[i], ensemble) != sign_classify(t[i], c)):
            incorrect += 1

    return float(incorrect) / float(X.shape[0])


def random_forest_classify(x, ensemble):
    vote_for_pos = 0
    vote_for_neg = 0
    #print("",len(ensemble))
    for j in range(len(ensemble)):
        if(ensemble[j].stump_classify(x) == 1):
            vote_for_pos += 1
        if(ensemble[j].stump_classify(x) == -1):
            vote_for_neg += 1
    if(vote_for_pos >= vote_for_neg):
        return 1
    else:
        return -1


def create_random_tree(X, t, m, features):
    feats = []
    for i in range(features):
        feats.append(i)

    rand_features = np.random.choice(np.asarray(feats), m, replace=False)
    high_ig = -1
    feature_split = -1
    for f in rand_features:
        ig = information_gain(X, t, f)
        if(high_ig < ig):
            high_ig = ig
            feature_split = f
    stump = Classifier(X, t, feature_split, 2)
    return stump


def get_bootstrap(X, t):
    data = np.append(X, t[np.newaxis].T, axis=1)
    mask = [0] * data.shape[0]
    for i in range(data.shape[0]):
        mask[random.randint(0, data.shape[0]-1)] = 1
    bagging = np.zeros((sum(mask), data.shape[1]))
    j = 0
    for i in range(len(mask)):
        if(mask[i] == 1):
            bagging[j] = data[i]
            j += 1
    return (bagging[:,:-1], bagging[:,-1])


def perform_random_forest(X, t, Xtest, ttest, m, iterations):
    ensemble = []
    errors = []
    test_errors = []
    for i in range(iterations):
        Xd, td = get_bootstrap(X, t)
        tree = create_random_tree(Xd, td, m, 9)
        ensemble.append(tree)
        errors.append(get_error(X, t, ensemble, 2))
        test_errors.append(get_error(Xtest, ttest, ensemble, 2))
    return (errors, test_errors)




def rf(file):
    X, t = get_dataset(file)
    X, t = remove_unlabelled_data(X, t)
    #remove patient ids
    X = X[:,1:]
    (Xtrain, ttrain, Xtest, ttest) = split_test_data_random(X, t, .9)

    err, terr = perform_random_forest(Xtrain, ttrain, Xtest, ttest, 3, 100)
    print("Training error rates for m=3:")
    for i in range(len(err)):
        print("    Iter ", i+1, ": ", err[i])
    print("Test error rates for m=3:")
    for i in range(len(terr)):
        print("    Iter ", i+1, ": ", terr[i])


    feature_nums = [2,3,4,5,6,7,8,9]
    for m in feature_nums:
        (Xtrain, ttrain, Xtest, ttest) = split_test_data_random(X, t, .9)
        err, terr = perform_random_forest(Xtrain, ttrain, Xtest, ttest, m, 100)
        print("RF choosing", m, "features")
        print("\tFinal train error:", err[-1])
        print("\tFinal test error:", terr[-1])

if(len(sys.argv) < 2):
    print("Use: rf.py <filename>")
    exit(0)
rf(sys.argv[1])
