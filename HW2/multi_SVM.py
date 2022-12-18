import math
import numpy as np
import sys
from cvxopt import solvers
from cvxopt import matrix as cmat
import random


###########################################
#                                         #
#           Dataset Functions             #
#                                         #
###########################################

def get_dataset_from_csv(file):
    DB = np.genfromtxt(file,delimiter=',')
    X=DB[:,0:-1]
    t=DB[:,-1]
    return (X,t)

def get_classes(target):
    classes = []
    for i in range(target.shape[0]):
        if(not target[i] in classes):
            classes.append(target[i])
    return sorted(classes)

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


def shuffle(X,t):
    newX = np.zeros(X.shape)
    newT = np.zeros(t.shape)
    indices = list(range(X.shape[0]))
    count = 0
    while len(indices) > 0:
        draw = random.choice(indices)
        indices.remove(draw)
        newX[count] = X[draw]
        newT[count] = t[draw]
        count = count + 1
    return (newX, newT)



#takes target vector and sends target values associated with 'class_one' to 1 and all others to -1
def target_to_sign(target, class_one):
    new_target = np.ones(target.shape[0])
    for i in range(target.shape[0]):
        if(target[i] != class_one):
            new_target[i] = -1

    return new_target

def get_mfeat_target():
    target = np.zeros(2000)
    digit = -1
    for i in range(2000):
        if i % 200 == 0:
            digit = digit + 1
        target[i] = digit
    return target

def get_dataset_from_mfeat(file):
    DB = np.genfromtxt(file)
    t = get_mfeat_target()
    return (DB, t)


def get_all_mfeat():
    filearr=['./mfeat/mfeat-fou', './mfeat/mfeat-fac', './mfeat/mfeat-kar', './mfeat/mfeat-pix', './mfeat/mfeat-zer', './mfeat/mfeat-mor']
    dataset = None
    target = None
    for file in filearr:
        (X,target) = get_dataset_from_mfeat(file)
        if(dataset is None):
            dataset = X
        else:
            dataset = np.concatenate((dataset,X), axis=1)
    return (dataset,target)



###########################################
#                                         #
#       MISCELLANEOUS AND KERNELS         #
#                                         #
###########################################

def write_stats(filename , rbfstats):
    f = open(filename, "w")
    for i in range(rbfstats.shape[0]):
        for j in range(rbfstats.shape[1]):
            print("Reg ", rbfstats[i,j,0], " HP ", rbfstats[i,j,1], " Avg ", rbfstats[i,j,2], " Var ", rbfstats[i,j,3])
    f.close()

def get_optimal_reg(stats):
    index = -1
    highest = 2
    for i in range(len(stats)):
        if(stats[i, 1]<highest):
            highest = stats[i,1]
            index = i
    return index


def get_optimal_reg_and_var(stats):
    index = (-1,-1)
    highest = 2
    for i in range(stats.shape[0]):
        for j in range(stats.shape[1]):
            if(stats[i,j]<highest):
                highest = stats[i,j,1]
                index = (i,j)
    return index


def sign(f):
    return f / abs(f)


def calculate_stats(avgs):
    average = 0
    for i in range(len(avgs)):
        average = average + avgs[i]

    average = average / float(len(avgs))
    dev = 0
    for i in range(len(avgs)):
        dev = dev + (avgs[i] - average) ** 2
    dev = dev / float(len(avgs))
    return (average, dev)


def get_optimal_reg(stats):
    index = -1
    highest = 2
    for i in range(len(stats)):
        if(stats[i, 1]<highest):
            highest = stats[i,1]
            index = i
    return index


def get_optimal_reg_and_var(stats):
    index = (-1,-1)
    lowest = 2
    for i in range(stats.shape[0]):
        for j in range(stats.shape[1]):
            if(stats[i,j,2] < lowest):
                lowest = stats[i,j,2]
                index = (i,j)

    return index



def ker_linear(x1,x2):
    return np.dot(x1,x2)

def ker_rbf(hyper):
    return lambda x1, x2: ker_rbf_math(x1,x2,hyper)

def ker_rbf_math(x1,x2, variance):
    return math.exp((np.linalg.norm(x1-x2) ** 2) / (-2 * variance))





###########################################
#                                         #
#              SVM FUNCTIONS              #
#                                         #
###########################################

def lagrangian_quadratic_form(data, target, kernel):
    datasize = data.shape[0]
    lqf = np.zeros([datasize, datasize])
    for i in range(datasize):
        for j in range(datasize):
            lqf[i,j] = target[i]*target[j]*kernel(data[i],data[j])

    return lqf

def target_matrix(target):
    targetsize = target.shape[0]
    target_mat = np.zeros(targetsize)
    for i in range(targetsize):
        target_mat[i,i] = target[i]
    return target_mat

def dual_optimizer(data, target, c, kernel):
    datasize = data.shape[0]
    featuresize = data.shape[1]

    lqf = lagrangian_quadratic_form(data, target, kernel)

    #Matrices for the Lagrangian function
    P = cmat(lqf)
    q = cmat(-1 * np.ones(datasize))

    #Box constraints on the Lagrangian vector
    G = cmat(np.identity(datasize))
    h = cmat(c * np.ones(datasize))

    #Linear constraints on Lagrangian vector
    A = cmat(target,(1,datasize),'d')
    b = cmat(0, (1,1), 'd')


    opt = solvers.qp(P, q, G, h, A, b)

    return opt['x']

def get_support_vectors(data, target, lagrange):
    count = 0
    for i in range(data.shape[0]):
        if(lagrange[i] != 0):
            count += 1
    support = np.zeros([count, data.shape[1]])
    support_target = np.zeros(count)
    slack = np.zeros(count)
    count = 0
    for i in range(data.shape[0]):
        if(lagrange[i] != 0):
            support[count] = data[i]
            support_target[count] = target[i]
            slack[count] = lagrange[i]
            count = count + 1
    return (support, support_target, slack)

def get_bias(support_vectors, support_targets, c,slack, kernel):
    b = 0
    (mX, mt) = get_not_max_support_vectors(support_vectors, support_targets, c, slack)
    for i in range(mX.shape[0]):
        temp = 0
        for j in range(support_vectors.shape[0]):
            temp = temp + (slack[j] * support_targets[j] * kernel(mX[i], support_vectors[j]))
        b = b + support_targets[i] - temp
    return (1 / float(mX.shape[0])) * b

def classify(support_vectors, support_targets, slack, bias, kernel, input):
    sum = 0
    for i in range(support_vectors.shape[0]):
        sum = sum + slack[i] * support_targets[i] * kernel(support_vectors[i], input)
    return sign(sum + bias)


def classify_raw(support_vectors, support_targets, slack, bias, kernel, input):
    sum = 0
    for i in range(support_vectors.shape[0]):
        sum = sum + slack[i] * support_targets[i] * kernel(support_vectors[i], input)
    return sum + bias


def classify_multi(multi_model, ker, vec):
    belief = [0] * len(multi_model)
    for mod in multi_model:
        belief[int(mod[0])] = classify_raw(mod[1], mod[2], mod[3], mod[4], ker, vec)

    index = -1
    highest = -1000000
    for i in range(len(belief)):
        if(belief[i] > highest):
            highest = belief[i]
            index = i
    return index


def test_multi_model(X_val, t_val, model, ker):
    correct = 0
    for test in range(X_val.shape[0]):
        test_vec = X_val[test]
        prediction = classify_multi(model, ker, test_vec)
        if(prediction == t_val[test]):
            correct = correct + 1
    return (correct, X_val.shape[0])


def train_multi_model(X, t, c, ker, classes, silent=True):
    multi_model = []
    for cl in classes:
        tg = target_to_sign(t, cl)
        lagrange_mult = dual_optimizer(X, tg, c, ker)
        (support_vectors, support_target, slack) = get_support_vectors(X, tg, lagrange_mult)
        bias = get_bias(support_vectors, support_target, c, slack, ker)

        oaa_model = [cl, support_vectors, support_target, slack, bias, ker_linear]
        multi_model.append(oaa_model)
        if(not silent):
            print("    Done with ", cl)
    return multi_model

def get_not_max_support_vectors(support_vectors, support_target, c, lagrange):
    count = 0
    for i in range(lagrange.shape[0]):
        if(lagrange[i]!=c):
            count = count + 1
    mX = np.zeros([count, support_vectors.shape[1]])
    mt = np.zeros(count)
    count = 0
    for i in range(lagrange.shape[0]):
        if(lagrange[i]!=c):
            mX[count] = support_vectors[i]
            mt[count] = support_target[i]
            count = count + 1
    return (mX, mt)












def multi_SVM(file, silent=True):
    solvers.options['show_progress'] = False   #silence optimizer progress report
    if(file.lower() == "all"):
        (X,t)=get_all_mfeat()
    else:
        (X,t)=get_dataset_from_mfeat(file)
    (X,t)=shuffle(X,t)

    C=[.000000001,.000001, .001, 1, 100]
    #(X,t)=(X[0:400],t[0:400])          #20 percent data
    (X_trainval, t_trainval, X_test, t_test) = get_80_20_split(X,t)
    foldsize = 10
    linear_stats = np.zeros([len(C), 3])
    count = 0
    classes = get_classes(t)

    #
    #linear kernel
    #
    count = 0
    for reg in C:
        averages = np.zeros(foldsize)
        for fold in range(foldsize):
            (X_train, t_train, X_val, t_val) = get_fold(fold, foldsize, X_trainval, t_trainval)
            multi_model = train_multi_model(X_train, t_train, reg, ker_linear, classes)

            (correct, total) = test_multi_model(X_val, t_val, multi_model, ker_linear)

            if(not silent):
                print("------------------------------")
                print("Regularizer", reg, "Fold: ",fold+1, ", Correct: ", correct, "/", X_val.shape[0])

            averages[fold] = 1-(correct / X_val.shape[0])

        stats = calculate_stats(averages)

        linear_stats[count, 0] = reg
        linear_stats[count, 1] = np.asarray(stats[0])
        linear_stats[count, 2] = np.asarray(stats[1])
        count = count + 1
        print("Reg: ", reg, " Average: ", stats[0], " Variance: ", stats[1])

    opt_reg = get_optimal_reg(linear_stats)
    model = train_multi_model(X_trainval, t_trainval, opt_reg, ker_linear, classes)
    (correct, total) = test_multi_model(X_test, t_test, model, ker_linear)
    print("Linear Kernel")
    print("Optimal Regularizer: ", C[opt_reg], " Error Rate: ", 1-(correct / total))
    print()

    #
    # rbf kernel
    #
    hyperparams = [.000000001,.001,.1, 10, 1000]
    rbf_stats = np.zeros([len(C),len(hyperparams), 4])
    count = 0
    for reg in C:
        param_index = 0
        for param in hyperparams:
            averages = np.zeros(foldsize)
            for fold in range(foldsize):
                (X_train, t_train, X_val, t_val) = get_fold(fold, foldsize, X_trainval, t_trainval)
                multi_model = train_multi_model(X_train, t_train, reg, ker_rbf(param), classes)

                (correct, total) = test_multi_model(X_val, t_val, multi_model, ker_rbf(param))
                if(not silent):
                    print("------------------------------")
                    print("Reglarizer:", reg, "Hyperparameter:", param, "Fold: ",fold+1, ", Correct: ", correct, "/", X_val.shape[0])
                averages[fold] = 1 - (correct / X_val.shape[0])

            stats = calculate_stats(averages)
            rbf_stats[count, param_index, 0] = reg
            rbf_stats[count, param_index, 1] = hyperparams[param_index]
            rbf_stats[count, param_index, 2] = stats[0]
            rbf_stats[count, param_index, 3] = stats[1]
            print("Reg.: ", reg, "Hyperparameter", hyperparams[param_index], " Avg. Error: ", stats[0], " Var.: ", stats[1])
            param_index = param_index + 1
        count = count + 1

    (opt_reg, opt_var)= get_optimal_reg_and_var(rbf_stats)
    model = train_multi_model(X_trainval, t_trainval, C[opt_reg], ker_rbf(hyperparams[opt_var]), classes)
    (correct, total) = test_multi_model(X_test, t_test, model, ker_rbf(hyperparams[opt_var]))
    print("RBF Kernel:")
    print("Opt. Reg.: ", C[opt_reg], " Opt. Var.: ", hyperparams[opt_var], " Error Rate: ", 1-(correct / total))
    print()

multi_SVM(sys.argv[1])
