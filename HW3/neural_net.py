import tensorflow as tf
import numpy as np

mnist = tf.keras.datasets.mnist
(X_train, t_train), (X_test, t_test) = mnist.load_data()


def get_one_hot(target, classes):
    new_target = np.zeros((target.shape[0], classes))
    for i in range(target.shape[0]):
        new_target[i][target[i]] = 1
    return new_target



def neural_net():
    epoch_count = 30
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    t_train_oh = get_one_hot(t_train, 10)
    t_test_oh = get_one_hot(t_test, 10)
    

    #create the model based on the hw specifications: 28x28 input, 128 hidden layer, 10 output
    net = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    net.compile(
        optimizer = 'SGD',
        loss = tf.keras.losses.CategoricalCrossentropy(),
        metrics=[tf.keras.metrics.CategoricalAccuracy()]
    )

    print("Fitting the model.")
    history = net.fit(X_train_norm, t_train_oh, batch_size=32, epochs = epoch_count, verbose=0)

    print("Evaluating the model.")
    loss, accuracy = net.evaluate(X_test_norm, t_test_oh, verbose = 0)

    
    losses = history.history['loss']
    accuracies = history.history['categorical_accuracy']

    for i in range(epoch_count):
        print("Epoch: ", i+1)
        print("\tLoss: ", losses[i])
        print("\tAccuracy: ", accuracies[i])

    print("Training set:")
    print("\tLoss: ", loss)
    print("\tAccuracy: ", accuracy)


    

def print_for_chart(loss, accuracy):
    print("losses")
    for l in loss:
        print(l)
    print("accuracies")
    for a in accuracy:
        print(a)
        


neural_net()
