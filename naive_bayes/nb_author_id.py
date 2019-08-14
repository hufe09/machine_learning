#!/usr/bin/python

""" 
    This is the code to accompany the Naive Bayes mini-project.

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""

from sklearn.naive_bayes import GaussianNB
from time import time
from collections import Counter
import sys
sys.path.append("../tools/")
from email_preprocess import preprocess


# features_train and features_test are the features for the training
# and testing dataset, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()


# your code goes here

t0 = time()
clf = GaussianNB()
clf_model = clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t1, 3), "s")

print("准确率", clf.score(features_test, labels_test))
print(clf_model)

# 统计预测值次数
print(Counter(pred).items())
