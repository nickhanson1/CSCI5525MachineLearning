import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
mnist = tf.keras.datasets.mnist
(X_train, t_train), (X_test, t_test) = mnist.load_data()


def get_one_hot(target, classes):
    new_target = np.zeros((target.shape[0], classes))
    for i in range(target.shape[0]):
        new_target[i][target[i]] = 1
    return new_target


def print_convergence_stats(lists, names):
    for i in range(len(lists)):
        batches = lists[i]
        print(names[i] + str(": "))
        for b in batches.keys():
            print("\tBatch Size: ", b)
            print("\tRuntime: ", len(batches[b].history['loss']))
            print("")
        print("------------------")
        print("")

            
def print_train_stats(epochs, hist):
    print("SGD, Batch Size 32")
        
    for i in range(epochs):
       print("Epoch ", i+1)
       print("\tLoss: ", (hist[0].history['loss'])[i])
       print("\tAccuracy: ", (hist[0].history['categorical_accuracy'])[i])
       print("")

    print("Training Set: ")
    print("\tLoss: ", hist[1])
    print("\tAccuracy: ", hist[2])


def print_for_chart(loss, accuracy):
    print("losses")
    for l in loss:
        print(l)
    print("accuracies")
    for a in accuracy:
        print(a)

def cnn_run():
    epoch_count = 50
    X_train_norm = X_train / 255.0
    t_train_oh = get_one_hot(t_train, 10)
    X_test_norm = X_test / 255.0
    t_test_oh = get_one_hot(t_test, 10)


    X_train_norm = X_train_norm[:,:,:,np.newaxis]
    X_test_norm = X_test_norm[:,:,:,np.newaxis]
    

    convergence = tf.keras.callbacks.EarlyStopping(patience = 2)
    
    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Conv2D(1, (3,3), activation='relu',\
            padding='same', input_shape=(28,28,1)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Dropout(.5))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(128, activation='relu'))
    cnn.add(layers.Dropout(.5))
    cnn.add(layers.Dense(10,activation='softmax'))

    cnn.compile(
        optimizer = 'SGD',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    
    history = cnn.fit(X_train_norm, t_train_oh, \
                      batch_size = 32, epochs = epoch_count, verbose=0)

    loss, accuracy = cnn.evaluate(X_test_norm, t_test_oh, verbose = 0)

    print_train_stats(epoch_count, (history, loss, accuracy))


def cnn():
    print("Training initial network.")
    cnn_run()
    SGD_info = {}
    Adam_info = {}
    Adagrad_info = {}

    batches = [32,64,96,128]
    
    for b in batches:
        SGD_info[b] = cnn_converge(b, 'SGD')
        print("SGD with size ", b, " mini batches completed.")
        Adam_info[b] = cnn_converge(b, 'adam')
        print("Adam with size ", b, " mini batches completed.")
        Adagrad_info[b] = cnn_converge(b, 'adagrad')
        print("Adagrad with size ", b, " mini batches completed.")

    runtime_stats = [SGD_info, Adam_info, Adagrad_info]
    
    print_convergence_stats(runtime_stats, ["SGD", "Adam", "Adagrad"])



def cnn_converge(minibatch, opt):
    X_train_norm = X_train / 255.0
    t_train_oh = get_one_hot(t_train, 10)
    X_test_norm = X_test / 255.0
    t_test_oh = get_one_hot(t_test, 10)


    X_train_norm = X_train_norm[:,:,:,np.newaxis]
    X_test_norm = X_test_norm[:,:,:,np.newaxis]
    

    convergence = tf.keras.callbacks.EarlyStopping(patience = 2, monitor='loss')
    
    cnn = tf.keras.Sequential()
    cnn.add(tf.keras.layers.Conv2D(1, (3,3), activation='relu',\
            padding='same', input_shape=(28,28,1)))
    cnn.add(layers.MaxPooling2D((2,2)))
    cnn.add(layers.Dropout(.5))
    cnn.add(layers.Flatten())
    cnn.add(layers.Dense(128, activation='relu'))
    cnn.add(layers.Dropout(.5))
    cnn.add(layers.Dense(10,activation='softmax'))

    cnn.compile(
        optimizer = opt,
        loss = tf.keras.losses.CategoricalCrossentropy()
    )

    
    history = cnn.fit(X_train_norm, t_train_oh, callbacks=convergence, \
                      batch_size = minibatch, epochs = 150, verbose=0)

    return history

    
cnn()
