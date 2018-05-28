import pandas as pd
from ast import literal_eval
import numpy as np

from KNN import kNearestNeighbor

def CSV_Producer(tripIDs, predicted_categories):
    d = {'Test_Trip_ID': pd.Series(tripIDs), 'Predicted_JourneyPatternID': pd.Series(predicted_categories)}
    df = pd.DataFrame(d)
    df.to_csv('testSet_JourneyPatternIDs.csv', sep='\t', index=False, columns=['Test_Trip_ID', 'Predicted_JourneyPatternID'])


trainSet = pd.read_csv('../../datasets/project_2/train_set.csv', converters={"Trajectory": literal_eval})
#trainSet = trainSet[:100]

testSet = pd.read_csv('../../datasets/project_2/test_set_a2.csv', sep="\t", converters={"Trajectory": literal_eval})['Trajectory']
tripIDs = list(range(1, 6))

X = []
y = []
test = []
for i in range(len(trainSet)):
    x = np.array(trainSet['Trajectory'][i])
    X.append(x)
    y.append(trainSet['journeyPatternId'][i])

for i in range(len(testSet)):
    x = np.array(testSet[i])
    test.append(x)

X = np.array(X)
y = np.array(y)
predictions = []
kNearestNeighbor(X, y, test, predictions, 5)
print(predictions)
print(tripIDs)

CSV_Producer(tripIDs, predictions)
