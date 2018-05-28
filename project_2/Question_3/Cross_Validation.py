import pandas as pd
from ast import literal_eval
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from KNN import kNearestNeighbor

X = []
y = []
skf = StratifiedKFold(n_splits=10)
trainSet = pd.read_csv('../../datasets/project_2/train_set.csv', converters={"Trajectory": literal_eval})
trainSet = trainSet[:int(len(trainSet['Trajectory'])*0.1)]

for i in range(len(trainSet)):
    x = np.array(trainSet['Trajectory'][i])
    X.append(x)
    y.append(trainSet['journeyPatternId'][i])

X = np.array(X)
y = np.array(y)

skf.get_n_splits(X, y)
accuracy = 0
for train_index, test_index in skf.split(X, y):
    X_train, index_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    predictions = []

    print("CRV\n")
    kNearestNeighbor(X_train, y_train, index_test, predictions, 5)

    accuracy += metrics.accuracy_score(y_test, predictions)
    print(accuracy)
    print("\n\n")

print("Accuracy: ", accuracy/10)








'''

Accuracy:  0.9249991539684694




from dtw import dtw
import sys
sys.path.insert(0, '..')
from HarvesineDistance import haversine
from sklearn.neighbors import KNeighborsClassifier
X = []
y = []


def Calculate_Distance(train, test):
    t_train = (train[0], train[0:])
    t_test = (test[0], test[0:])
    return dtw(t_train[1], t_test[1], dist=lambda spot1, spot2: haversine(X[t_train[0]][spot1][1], X[t_train[0]][spot1][2],
     X[t_test[0]][spot2][1], X[t_test[0]][spot2][2])[0]
     # PROVLHMA exei allajei h prwth timh tou train opou edeixne se poia grammh tou X koitame 
      
dist = Calculate_Distance
knn = KNeighborsClassifier(n_neighbors=5, metric=dist)
skf = StratifiedKFold(n_splits=10)

trainSet = pd.read_csv('../../datasets/project_2/train_set.csv', converters={"Trajectory": literal_eval})
trainSet = trainSet[:100]



index = []
max_v = 0
for i in range(len(trainSet)):
    x = np.array(trainSet['Trajectory'][i])
    if x.shape[0] > max_v:
        max_v = x.shape[0]
    X.append(x)
    y.append(trainSet['journeyPatternId'][i])
    
    
# dhmiourgoume ena array pou periexei arraies opou h prwth tou
# timh tha deixnei se poia seira tou X koitame kai ta upoloipa
# tha einai apo to 0 - max_v kai tha dhlwnoun se poia tripleta 
# tou X koitame... tha prepei na ginetai elegxos mhn h grammh 
# periexei ligoteres apo max_v tripletes
for i in range(len(X)):
    li = [i]
    li = li +list(range(max_v))
    index.append(np.array(li))



index = np.array(index)
X = np.array(X)
y = np.array(y)

skf.get_n_splits(index, y)
accuracy = 0
for train_index, test_index in skf.split(index, y):
    I_train, I_test = index[train_index], index[test_index]
    y_train, y_test = y[train_index], y[test_index]
    predictions = []
    knn.fit(I_train, y_train)
    print("predict")
    predictions = knn.predict(I_test)
    accuracy += metrics.accuracy_score(y_test, predictions)
    print(accuracy)
    print("\n\n")

print("Accuracy: ", accuracy/10)'''

