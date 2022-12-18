
from PIL import Image
import random
import math
import numpy as np
import sys
import os

###########################
#    UTILITY FUNCTIONS    #
###########################

def get_picture(file):
    img = Image.open(file).convert("RGB")
    arr = np.array(img)
    return arr


def norm(vec):
    norm = 0
    for i in range(vec.shape[0]):
        norm += vec[i] ** 2
    return math.sqrt(norm)



# for easily getting data for the plots
def print_raw_data(obj):
    for i in range(len(obj)):
        print(obj[i])



###########################
#    K-MEANS FUNCTIONS    #
###########################

def objective(X, means, indicators):
    obj = 0
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            for n in range(len(means)):
                obj += indicators[x,y,n] * (norm(X[x,y]-means[n]) ** 2)

    return obj


def get_initial_prototypes(k):
    prototypes = []
    for i in range(k):
        p = np.asarray(
            [random.randint(0,255),random.randint(0,255),random.randint(0,255)])
        prototypes.append(p)
    return prototypes

def get_class(x, y, indicators):
    for i in range(indicators.shape[2]):
        if(indicators[x,y,i] == 1):
            return i
    return -1

def generate_segmented_image(X, means, indicators):
    segimg = np.zeros((X.shape[0], X.shape[1], 3))
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            kclass = get_class(x, y, indicators)
            segimg[x,y] = means[kclass]
    return segimg


def update_indicators(X, means):
    indicators = np.zeros((X.shape[0], X.shape[1], len(means)))
    for x in range(X.shape[0]):
        for y in range(X.shape[1]):
            mindist = 1000000.0
            index = -1
            for i in range(len(means)):
                dist = norm(X[x,y] - means[i])
                if(dist < mindist):
                    index = i
                    mindist = dist
            indicators[x,y,index] = 1
    return indicators

def update_means(X, k, indicators):
    means = []
    for i in range(k):
        class_size = 0
        vec = np.asarray([0,0,0])
        for x in range(X.shape[0]):
            for y in range(X.shape[1]):
                if(indicators[x,y,i] == 1):
                    class_size += 1
                    vec += X[x,y]
        for j in range(vec.shape[0]):
            if(class_size == 0):
                vec[j] = 0
            else:
                vec[j] = int(float(vec[j]) / float(class_size))
        means.append(vec)
    return means

def perform_k_means(X, k, iterations):
    means = get_initial_prototypes(k)
    indicators = None
    obj_vals = []
    for i in range(iterations):
        indicators = update_indicators(X, means)
        means = update_means(X, k, indicators)
        obj_vals.append(objective(X, means, indicators))

    return (generate_segmented_image(X, means, indicators), obj_vals)



def convert_and_save_image(imgarr, loc, name):
    finimg = Image.fromarray(np.uint8(imgarr), 'RGB')
    finimg.save(loc + name)






def kmeans(file):
    loc = "segmented_images/"
    try:
        os.mkdir("segmented_images")
    except FileExistsError:
        pass
    photoname = file.split('.')[0]
    X = get_picture(file)
    obj_measures = {}
    iterations = 15
    K = [3,5,7]
    for k in K:
        print("Beginning k-means with k =", k, "with", iterations, "iterations.")
        (imgarr, vals) = perform_k_means(X, k, iterations)
        obj_measures[k] = vals

        convert_and_save_image(imgarr, loc, photoname + "_seg" + str(k) + ".png")

    print()
    for k in K:
        print("Objective function values for k =", k)
        for i in range(len(obj_measures[k])):
            print("\tIteration ", i, ": ", obj_measures[k][i])
        print("")

if(len(sys.argv) < 2):
    print("Use: kmeans.py <filename>")
    exit(0)

kmeans(sys.argv[1])
