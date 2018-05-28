from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn import svm
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import numpy as np


def testSet_categoriesCSV(predicted_categories, ids):
    d = {'ID': pd.Series(ids), 'Predicted_Category': pd.Series(predicted_categories)}
    df = pd.DataFrame(d)
    df.to_csv('Produced_Files/testSet_categories.csv', sep='\t', index=False, columns=['ID', 'Predicted_Category'])


size = 10000
components = 160
my_additional_stop_words = ['people', 'said', 'did', 'say', 'says', 'year', 'day', 'just', 'good', 'come', 'make',
                            'going', 'having', 'like', 'need', 'given', 'got']
vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS.union(my_additional_stop_words))
le = preprocessing.LabelEncoder()
lsi_model = TruncatedSVD(n_components=components)
ps = PorterStemmer()
lmtzr = WordNetLemmatizer()

clf = svm.SVC(kernel='rbf', C=1, gamma=1)
# clf = SGDClassifier()

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------TRAIN-------------------------------------------------------------

dataset = pd.read_csv('../datasets/project_1/train_set.csv', sep="\t")
#dataset = dataset[0:size]
le.fit(dataset["Category"])
y = le.transform(dataset["Category"])
X = dataset['Content'] + 10*dataset['Title']
dataset = None

# preprocessing the data -- stemming or Lemmatization of data
new_X = []
for x in X:
    new_x = [""]
    for word in x.split(" "):
        # new_word = re.sub(",|’|\.|“|\"", "", lmtzr.lemmatize(word.lower()))
        new_word = ps.stem(re.sub(",|’|\.|“|\"|\?|!", "", word.lower()))   # decode('utf-8')
        new_x[0] = new_x[0] + " " + new_word
    new_X.append(new_x[0])
X = pd.Series(new_X)
new_X = None
print("End of preprocessing")


# vectorization of data
X = vectorizer.fit_transform(X).toarray()

# more processing the data -- doubling top values
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):
        if X[i][j] > 0.5:
            X[i][j] = X[i][j]*2
print("End of Vectorization")

X = lsi_model.fit_transform(X)
print("end of LSI")


# train the clf
clf.fit(X, y)
print("End of TRAIN")

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------TEST-----------------------------------------------------------

# predictions
predictions = []
dataset = pd.read_csv('../datasets/project_1/test_set.csv', sep="\t")
# dataset = dataset[0:size]
X = dataset['Content'] + 10*dataset['Title']

# preprocessing the data -- stemming or Lemmatization of data
new_X = []
for x in X:
    new_x = [""]
    for word in x.split(" "):
        # new_word = re.sub(",|’|\.|“|\"", "", lmtzr.lemmatize(word.lower()))
        new_word = ps.stem(re.sub(",|’|\.|“|\"|\?|!", "", word.lower()))  # decode('utf-8')
        new_x[0] = new_x[0] + " " + new_word
    new_X.append(new_x[0])
X = pd.Series(new_X)
new_X = None
print("End of preprocessing")

X = vectorizer.fit_transform(X).toarray()
# more processing the data -- doubling top values
for i in range(0, X.shape[0]):
    for j in range(0, X.shape[1]):
        if X[i][j] > 0.5:
            X[i][j] = X[i][j]*2
print("End of Vectorization")


X = lsi_model.fit_transform(X)

predictions = clf.predict(X)
predictions = le.inverse_transform(predictions)
testSet_categoriesCSV(predictions, dataset['Id'])
