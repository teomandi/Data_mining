from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


train_data = pd.read_csv('../datasets/project_1/train_set.csv', sep="\t")

le = preprocessing.LabelEncoder()
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])

# vectorization of data
my_additional_stop_words = ['people', 'said', 'did', 'say', 'says', 'year', 'day', 'just', 'good', 'come', 'make', 'going', 'having', 'like', 'need', 'given', 'got']
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS.union(my_additional_stop_words))
X = vectorizer.fit_transform(train_data['Content'] + 10*train_data['Title']).toarray()

components = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400]
accuracy = []

choice = 1
clf_str = ""
if choice == 1:
    clf_str = "SVM"
    clf = svm.SVC(kernel='rbf', C=1, gamma=1)
else:
    clf_str = "SGDC"
    clf = SGDClassifier()

for component in components:
    print(component)
    lsi_model = TruncatedSVD(n_components=component)
    lsi_X = lsi_model.fit_transform(X)

    skf = StratifiedKFold(n_splits=10)
    skf.get_n_splits(lsi_X, y)
    clf_accuracy = 0
    for train_index, test_index in skf.split(lsi_X, y):
        predictions = []
        X_train, X_test = lsi_X[train_index], lsi_X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        clf_accuracy += metrics.accuracy_score(y_test, predictions)

    accuracy.append(clf_accuracy/10)

plt.plot(components, accuracy)
plt.ylabel(clf_str+' accuracy')
plt.xlabel('Components')
plt.show()



