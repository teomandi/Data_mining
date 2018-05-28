import numpy as np
from collections import Counter


def predict(x_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(x_train)):
        # first we compute the euclidean distance
        distance = np.sqrt(np.sum(np.square(x_test - x_train[i, :])))
        # add it to list of distances
        distances.append([distance, i])

    # sort the list
    distances = sorted(distances)

    # make a list of the k neighbors' targets
    for i in range(k):
        index = distances[i][1]
        targets.append(y_train[index])

    # return most common target..perform a majority vote using a Counter.
    return Counter(targets).most_common(1)[0][0]


def kNearestNeighbor(x_train, y_train, x_test, predictions, k):
    # loop over all observations
    for i in range(len(x_test)):
        predictions.append(predict(x_train, y_train, x_test[i, :], k))
