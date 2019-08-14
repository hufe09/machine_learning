#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
import pprint

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson14_keys.pkl')
labels, features = targetFeatureSplit(data)

# your code goes here
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3,
                                                                            random_state=42)
# it's all yours from here forward!
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
print(clf.score(features_test, labels_test))

print("Number of people in test set is", len(labels_test))

count = 0
for i in labels_test:
    if i == 1:
        count += 1

print("Number of POIs in test set is", count)


def evaluation_metrics(truth, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(truth)):
        if truth[i] == 1 and pred[i] == 1:
            TP += 1
        if truth[i] == 1 and pred[i] == 0:
            FN += 1
        if truth[i] == 0 and pred[i] == 1:
            FP += 1
        if truth[i] == 0 and pred[i] == 0:
            TN += 1
    model_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "F1": None,
        "F2": None
    }
    try:
        total = TP + TN + FP + FN
        print("Total", total, "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

        accuracy = 1.0 * (TP + TN) / total
        model_metrics["accuracy"] = accuracy
        precision = 1.0 * TP / (TP + FP)
        model_metrics["precision"] = precision
        recall = 1.0 * TP / (TP + FN)
        model_metrics["recall"] = recall
        # f1 = 2.0 * TP / (2 * TP + FP + FN)
        f1 = 2.0 * (precision * recall)/(precision + recall)
        model_metrics["F1"] = f1
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        model_metrics["F2"] = f2
    except:
        print("Got a divide by zero when trying out the set.")
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")

    return TP, TN, FP, FN, model_metrics


def sklearn_metrics(truth, prediction):
    from sklearn import metrics

    print("Precision score", metrics.precision_score(truth, prediction))
    print("Accuracy score", metrics.accuracy_score(truth, prediction))
    print("Recall score", metrics.recall_score(truth, prediction))
    print("F1 score", metrics.f1_score(truth, prediction))


TP, TN, FP, FN, model_metrics = evaluation_metrics(labels_test, pred)
pprint.pprint(model_metrics)
sklearn_metrics(labels_test, pred)


##########################################################################
print("#"*100)
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

TP, TN, FP, FN, model_metrics = evaluation_metrics(true_labels, predictions)

pprint.pprint(model_metrics)
sklearn_metrics(true_labels, predictions)
