#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
from collections import Counter
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# your code goes here
from sklearn import tree

t0 = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)
print(clf.fit(features_train, labels_train))

print("fitting time:", round(time()-t0, 3), "s")
pred = clf.predict(features_test)
# print(pred)

t1 = time()
print('准确率: ', clf.score(features_test, labels_test))
print("predicting time:", round(time()-t1, 3), "s")

# 统计预测值次数
print(Counter(pred).items())
