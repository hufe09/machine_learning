#!/usr/bin/python

""" 
    This is the code to accompany the SVM mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm
from collections import Counter

# features_train and features_test are the features for the training
# and testing dataset, respectively
# labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

# features_train = features_train[:int(len(features_train)/100)]
# labels_train = labels_train[:int(len(labels_train)/100)]
# your code goes here
t0 = time()
clf = svm.SVC(kernel="linear")
# clf = svm.SVC(kernel="rbf", gamma="auto", C=10000)
print(clf.fit(features_train, labels_train))

print("fitting time:", round(time()-t0, 3), "s")
pred = clf.predict(features_test)
# print(pred)

t1 = time()
print('准确率: ', clf.score(features_test, labels_test))
print("predicting time:", round(time()-t1, 3), "s")

# 统计预测值次数
print(Counter(pred).items())

# 预测某元素的值
print(f'测试集的元素为10:{pred[10]}, 测试集的元素为26:{pred[26]}, 测试集的元素为50:{pred[50]}')



