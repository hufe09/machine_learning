#!/usr/bin/python

import pickle
import numpy

numpy.random.seed(42)

# The words (features) and authors (labels), already largely processed.
# These files should have been created from the previous (Lesson 10)
# mini-project.
words_file = "../text_learning/your_word_data.pkl"
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load(open(words_file, "rb"))
authors = pickle.load(open(authors_file, "rb"))

# test_size is the percentage of events assigned to the test set (the
# remainder go into training)
# feature matrices changed to dense representations for compatibility with
# classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors,
                                                                            test_size=0.1,
                                                                            random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()
words_bag = vectorizer.get_feature_names()

# a classic way to overfit is to use a small number
# of data points and a large number of features;
# train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

print(f"训练集大小：{len(features_train)}")

# your code goes here
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

# accuracy_score method 1
score = clf.score(features_test, labels_test)
# accuracy_score method 2
acc = accuracy_score(pred, labels_test)
print(acc)

# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
feature_importance = clf.feature_importances_
print(f"特征重要性最大值：{max(feature_importance)}, 特征重要性最小值：{min(feature_importance)}")
print(f"该特征单词的数量：{feature_importance.argmax()}")

# 所有单词都同等重要，每个单词的重要性都低于 0.01
imp_list = []
for index, feature in enumerate(feature_importance):
    if feature > 0.2:
        imp_list.append(feature)
        print(f"单词词索引：{index}, 单词词数量：{feature}, 单词：{words_bag[index]}")
if not imp_list:
    print("无超过阈值0.2的特征重要性")

# 使用 TfIdf 获得最重要的单词
# print(f"获得对应的单词是：{words_bag[33614]}")  # sshacklensf
# print(f"获得对应的单词是：{words_bag[14343]}")  # cgermannsf
