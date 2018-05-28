from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn import preprocessing
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm

from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import pandas as pd
import numpy as np

import re
import time
from KNN_Classifier import kNearestNeighbor

from sklearn.model_selection import cross_val_predict

def EvaluateMetricCSV(accuracy, precision, recall, f1):

    d = {'Statistic Measure': pd.Series(["Accuracy", "Precision", "Recall", "F-Measure"]),
         'Naive Bayes': pd.Series([accuracy[0], precision[0], recall[0], f1[0]]),
         'Random Forest': pd.Series([accuracy[1], precision[1], recall[1], f1[1]]),
         'SVM': pd.Series([accuracy[2], precision[2], recall[2], f1[2]]),
         'KNN': pd.Series([accuracy[4], precision[4], recall[4], f1[4]]),
         'Stochastic Gradient Descent': pd.Series([accuracy[3], precision[3], recall[3], f1[3]])}
    df = pd.DataFrame(d)
    df.to_csv('Produced_Files/EvaluationMetric_10fold.csv', sep='\t', index=False,
            columns=['Statistic Measure', 'Naive Bayes', 'Random Forest', 'SVM', 'KNN', 'Stochastic Gradient Descent'])
    print("\n\n\nAccuracy\t", accuracy[0], "\t", accuracy[1], "\t", accuracy[2], "\t", accuracy[4], "\t", accuracy[3],
          "\n\nsize = ", size)


# initialization
size = 10000
components = 160
k = 30
my_additional_stop_words = ['people', 'said', 'did', 'say', 'says', 'year', 'day', 'just', 'good', 'come', 'make', 'going', 'having', 'like', 'need', 'given', 'got']
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS.union(my_additional_stop_words))
le = preprocessing.LabelEncoder()
lsi_model = TruncatedSVD(n_components=components)
ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

train_data = pd.read_csv('../datasets/project_1/train_set.csv', sep="\t")
#train_data = train_data[0:size]
X = train_data['Content'] + 10*train_data['Title']
le.fit(train_data["Category"])
y = le.transform(train_data["Category"])


# preprocessing the data -- stemming or Lemmatization of data
new_X = []
for x in X:
    new_x = [""]
    for word in x.split(" "):
        #new_word = re.sub(",|’|\.|“|\"", "", lmtzr.lemmatize(word.lower()))
        new_word = ps.stem(re.sub(",|’|\.|“|\"|\?|!", "", word.lower()))  # decode('utf-8')
        new_x[0] = new_x[0] + " " + new_word
    new_X.append(new_x[0])
X = pd.Series(new_X)
new_X = None
print("End of preprocessing")

# vectorization of data
X = vectorizer.fit_transform(X).toarray()


# more processing the data -- doubling top values
'''for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):
        if X[i][j] > 0.5:
            X[i][j] = X[i][j]*2'''

print("End of Vectorization")
predictions = []
start_time = time.time()
precision_score = []
recall_score = []
f1_score = []
accuracy_score = []

# lsi to data
print("LSI")
X = lsi_model.fit_transform(X)


# k-fold Cross Validation
skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X, y)
for i in range(0, 5):
    if i == 0:
        print("Naive Bayes")
    elif i == 1:
        print("Random Forest")
    elif i == 2:
        print("SVM")
    elif i == 3:
        print("SGDC")
    elif i == 4:
        print("KNN k = ", k)

    clf_precision = 0
    clf_recall = 0
    clf_f1 = 0
    clf_accuracy = 0
    for train_index, test_index in skf.split(X, y):
        predictions = []

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if (i < 4) and (i > 0):
            if i == 1:
                clf = RandomForestClassifier()
            elif i == 2:
                # parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [1, 10]}
                # svc = svm.SVC()
                # clf = GridSearchCV(svc, parameters)'''
                clf = svm.SVC(kernel='rbf', C=1, gamma=1)
            elif i == 3:
                clf = SGDClassifier()

            clf.fit(X_train, y_train)
            predictions = clf.predict(X_test)
        else:
            if i == 0:
                scaler = MinMaxScaler(feature_range=(1, 100))
                scaler.fit(X_train)
                scaled_train = scaler.transform(X_train)

                scaler.fit(X_test)
                scaled_test = scaler.transform(X_test)

                clf = MultinomialNB()
                clf.fit(scaled_train, y_train)
                predictions = cross_val_predict(clf, scaled_test, y_test)

            elif i == 4:
                kNearestNeighbor(X_train, y_train, X_test, predictions, k)

        predicted_categories = le.inverse_transform(predictions)
        clf_precision += metrics.precision_score(y_test, predictions, average='micro')
        clf_recall += metrics.recall_score(y_test, predictions, average='micro')
        clf_f1 += metrics.f1_score(y_test, predictions, average='micro')
        clf_accuracy += metrics.accuracy_score(y_test, predictions)

    # compute the average of each value
    precision_score.append(clf_precision/10)
    recall_score.append(clf_recall/10)
    f1_score.append(clf_f1/10)
    accuracy_score.append(clf_accuracy/10)

EvaluateMetricCSV(accuracy_score, precision_score, recall_score, f1_score)
print("Time %f" % (time.time() - start_time))
