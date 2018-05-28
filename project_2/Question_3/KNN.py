
from collections import Counter
from dtw import dtw
import sys
sys.path.insert(0, '..')
from HarversineDistance import harversine



def predict(x_train, y_train, x_test, k):
    # create list for distances and targets
    distances = []
    targets = []

    for i in range(len(x_train)):
        distance = dtw(x_train[i], x_test, dist=lambda spot1, spot2: harversine
                (spot1[1], spot1[2], spot2[1], spot2[2]))[0]

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
        predictions.append(predict(x_train, y_train, x_test[i], k))
