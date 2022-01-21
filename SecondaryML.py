from sklearn import svm
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

import time


f = open('motifSequences.pkl', 'rb')
array = pickle.load(f)

posSequences = array[0]
negSequences = array[1]

shape_file = open('shapeArrays.pkl', 'rb')
shapes = pickle.load(shape_file)
EP_shape = shapes[0]
HeIT = shapes[1]
MGW = shapes[2]
ProT = shapes[3]
Roll = shapes[4]

def createStartSets(posSequences, negSequences):
    dataSet = []
    labelSet = []
    for sequence in posSequences:
        labelSet.append(1)
        dataSet.append(convertToOneHot(sequence))

    for sequence in negSequences:
        labelSet.append(0)
        dataSet.append(convertToOneHot(sequence))

    # for i in range(len(dataSet)-1):
    #     #dataSet[i].append([EP_shape[i], HeIT[i], MGW[i], ProT[i], Roll[i]])
    #     dataSet[i] = np.append(dataSet[i], EP_shape[i])
    #     dataSet[i] = np.append(dataSet[i], HeIT[i])
    #     dataSet[i] = np.append(dataSet[i], MGW[i])
    #     dataSet[i] = np.append(dataSet[i], ProT[i])
    #     dataSet[i] = np.append(dataSet[i], Roll[i])


    return [dataSet, labelSet]

# start index, end index, chromosome, +/-
def convertToOneHot(sequence):
    # get sequence into an array
    seq_array = list(sequence)

    # integer encode the sequence
    label_encoder = LabelEncoder()
    integer_encoded_seq = label_encoder.fit_transform(seq_array)

    # one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False)
    # reshape
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)

    return onehot_encoded_seq

start_sets = createStartSets(posSequences, negSequences)

numpy_data = np.array(start_sets[0])
numpy_labels = np.array(start_sets[1])
numpy_data_size = len(numpy_data)
numpy_data = numpy_data.reshape(numpy_data_size,-1)
#
# print(numpy_data_size)
numpy_data_total = np.empty((13205,124))
#
for i in range(len(numpy_data)-1):
    numpy_data_total[i] = np.concatenate((numpy_data[i], EP_shape[i], HeIT[i], MGW[i], ProT[i], Roll[i]), axis=None)
#     #numpy_data_total[i] = np.append(numpy_data[i], EP_shape[i])
#     # numpy_data_total[i] = np.append(numpy_data[i], HeIT[i])
#     # numpy_data_total[i] = np.append(numpy_data[i], MGW[i])
#     # numpy_data_total[i] = np.append(numpy_data[i], ProT[i])
#     # numpy_data_total[i] = np.append(numpy_data[i], Roll[i])
#
#numpy_data_total_HeIT = np.empty((13205,7))

random_data = np.random.rand(13205,124)

X_train, X_test, y_train, y_test = train_test_split(numpy_data_total, numpy_labels, test_size=0.3, random_state=109)

start_sets[1] = start_sets[1][0:-1]

# X_train, X_test, y_train, y_test = train_test_split(HeIT, start_sets[1], test_size=0.3, random_state=109)

# numpy_X_train = np.array(X_train)
# numpy_X_test = np.array(X_test)
# numpy_Y_train = np.array(y_train)
#
# X_train_size = len(numpy_X_train)
# TwoDim_X_train = numpy_X_train.reshape(X_train_size,-1)
#
# X_test_size = len(numpy_X_test)
# TwoDim_X_test = numpy_X_test.reshape(X_test_size,-1)

# #Create a svm Classifier
start_SVM = time.time()
SVM_clf = svm.SVC(kernel='rbf') # Linear Kernel
SVM_clf.fit(X_train, y_train)
SVM_pred = SVM_clf.predict(X_test)
end_SVM = time.time()

print("SVM Run Time:", end_SVM - start_SVM)
print("SVM Accuracy:", metrics.accuracy_score(y_test, SVM_pred))
print("SVM Precision:", metrics.precision_score(y_test, SVM_pred))
# #
# Create a nearest neighbours classifier
start_KNear = time.time()
KNear_clf = KNeighborsClassifier(n_neighbors=3)
KNear_clf.fit(X_train, y_train)
nearest_pred = KNear_clf.predict(X_test)
end_KNear = time.time()

print("KNear Run Time:", end_KNear - start_KNear)
print("K-Nearest Accuracy:",metrics.accuracy_score(y_test, nearest_pred))
print("K-Nearest Precision:", metrics.precision_score(y_test, nearest_pred))

#Create a Gaussian Classifier
start_Gauss = time.time()
Bayes_clf = GaussianNB()
Bayes_clf.fit(X_train, y_train)
bayes_pred = Bayes_clf.predict(X_test)
end_Gauss = time.time()
print("Gauss Run Time:", end_Gauss - start_Gauss)
print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, bayes_pred))
print("Naive Bayes Precision:", metrics.precision_score(y_test, bayes_pred))

##Create a Logistic Regression Classifier
start_Logis = time.time()
Logist_clf = LogisticRegression(solver='lbfgs', max_iter=2000)
Logist_clf.fit(X_train, y_train)
logist_pred = Logist_clf.predict(X_test)
end_Logis = time.time()
print("Logistic Run Time:", end_Logis - start_Logis)
print("Logistic Regression Accuracy:", metrics.accuracy_score(y_test, logist_pred))
print("Logistic Regression Precision:", metrics.precision_score(y_test, logist_pred))

strat = ["most_frequent", "stratified", "constant", "uniform"]

for s in strat:
     if s == "constant":
             dummy_clf = DummyClassifier(strategy=s,random_state=None,constant=1)
     else:
             dummy_clf = DummyClassifier(strategy=s,random_state=None)
     dummy_clf.fit(X_train, y_train)
     dummy_pred = dummy_clf.predict(y_test)
     print(f"{s} Score Accuracy:", metrics.accuracy_score(y_test, dummy_pred))
     print("----------------------xxxxxxx----------------------")