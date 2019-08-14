#!/usr/bin/python

from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from time import time
from prep_terrain_data import makeTerrainData
from class_vis import pretty_picture
from sklearn import metrics
import matplotlib.pyplot as plt
import pprint
import pandas as pd

pd.set_option("display.max_columns", 100)

terrain_data = makeTerrainData()


def clf_model(clf, algorithm, data=terrain_data):
    print("*" * 50, algorithm, "*" * 50)
    features_train, labels_train, features_test, labels_test = data

    # the training data (features_train, labels_train) have both "fast" and "slow"
    # points mixed together--separate them so we can give them different colors
    # in the scatter plot and identify them visually
    grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
    bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 0]
    grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii] == 1]
    bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii] == 1]

    # initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color="b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color="r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.title(algorithm)
    plt.show()

    # your code here!  name your classifier object clf if you want the
    # visualization code (prettyPicture) to show you the decision boundary

    t0 = time()
    # clf = KNeighborsClassifier()
    print(clf)
    model = clf.fit(features_train, labels_train)
    fitting_time = round(time() - t0, 3)
    print("Fitting time:", fitting_time, "s")

    t1 = time()
    pred = clf.predict(features_test)
    predicting_time = round(time() - t1, 3)
    print("Predicting time:", predicting_time, "s")

    model_score = clf.score(features_test, labels_test)
    accuracy = round(metrics.accuracy_score(labels_test, pred), 5)
    precision = round(metrics.precision_score(labels_test, pred), 5)
    recall = round(metrics.recall_score(labels_test, pred), 5)
    F1 = round(metrics.f1_score(labels_test, pred), 5)

    print("score:", model_score, "\naccuracy:", accuracy, "\nprecision:", precision,
          "\nrecall:", recall, "\nF1:", F1)

    try:
        pretty_picture(clf, features_test, labels_test, algorithm)
    except NameError:
        pass
    return {"algorithm": algorithm, "model": str(model), "score": model_score,
            "fitting_time": fitting_time, "predicting_time": predicting_time,
            "accuracy": accuracy, "precision": precision, "recall": recall, "F1": F1}


if __name__ == "__main__":
    algorithms = [
        clf_model(GaussianNB(), "Naive Bayes"),
        clf_model(svm.SVC(gamma="auto"), "SVM"),
        clf_model(tree.DecisionTreeClassifier(), "Decision Tree"),
        clf_model(KNeighborsClassifier(), "KNN"),
        clf_model(RandomForestClassifier(), "Random Forest"),
        clf_model(AdaBoostClassifier(), "AdaBoost"),
    ]

    res = []
    for algorithm in algorithms:
        res.append(algorithm)

    pprint.pprint(res)

    df = pd.DataFrame(res, columns=["algorithm", "model", "score", "fitting_time",
                                    "predicting_time", "accuracy", "precision",
                                    "recall", "F1"]).drop("model", axis=1)
    print("根据F1值排序")
    print(df.sort_values(by="F1", ascending=False))
    print("根据训练时间排序")
    print(df.sort_values(by="fitting_time"))
