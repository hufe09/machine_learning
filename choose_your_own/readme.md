# 选择你自己的算法

任何数据分析师具备的一项关键技能就是不断从机器学习中得到新的认识，这也是本节课的学习目标。这节课的内容是一个迷你项目。目标是用你选择的算法来做地形分类，并由你自己进行研究和部署。

我们无法检查你的结果，因为你有太多的算法和参数组合可以尝试了，但是你看到过我们上一个算法（朴素贝叶斯、SVM、决策树）所得出的准确率，因此你可以自行评估新的算法是否更好。

可选的算法如下：

k nearest neighbors（k 最近邻 或 KNN）

random forest（随机森林）

adaboost（有时也叫“被提升的决策树”）

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.qbrx09k6wih.png)

## 集成方法

一般在很多决策树基础上建立的元分类器，有很多分类器，然后把它们集合起来，做出最后的决定。

# 无人驾驶车Stanley

运用监督学习训练无人驾驶车Stanley的速度，它需要在沙漠中调节自己的速度，需要根据人类可以观察的一些特征实现这个功能，这个例子中只选取两种特征，一个是地形坡度，也就是汽车在平地上还是山坡上；第二个特征值是地形的平整性，也就是可以在车内可以实际测量的汽车上下颠簸的情况。

这个例子中，通过监督机器学习的方式，帮助Stanley决定它该加速还是减速。

- 颠簸程度 Bumpy ：smooth/bad
- 坡度 Slope ：flat/steep

*choose_your_own/stanley_speed.py*

## [Naive Bayes](<https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes>)

```
from sklearn.naive_bayes import GaussianNB

clf = KNeighborsClassifier()
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
```

```
GaussianNB(priors=None, var_smoothing=1e-09)

Fitting time: 0.004 s
Predicting time: 0.0 s
score: 0.884 
accuracy: 0.884 
precision: 0.89143 
recall: 0.93976 
F1: 0.91496
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/Naive%20Bayes.75gnir3gc15.png)

## [SVM](<https://scikit-learn.org/stable/modules/svm.html#svm>)

```
from sklearn import svm


clf = svm.SVC(gamma="auto")
```

```
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

Fitting time: 0.007 s
Predicting time: 0.001 s
score: 0.92 
accuracy: 0.92 
precision: 0.94512 
recall: 0.93373 
F1: 0.93939
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/SVM.h94io6kxoec.png)

## [Decision Trees](<https://scikit-learn.org/stable/modules/tree.html#tree>)

```
from sklearn import tree


clf = tree.DecisionTreeClassifier()
```

```
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')

Fitting time: 0.002 s
Predicting time: 0.001 s
score: 0.908 
accuracy: 0.908 
precision: 0.91813 
recall: 0.94578 
F1: 0.93175
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/Decision%20Tree.jiojoii40d.png)

## [ KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)

k nearest neighbors，k 最近邻

```
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
```

```
KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')

Fitting time: 0.002 s
Predicting time: 0.019 s
score: 0.92 
accuracy: 0.92 
precision: 0.94512 
recall: 0.93373 
F1: 0.93939
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/KNN.5z5vmi5xhsq.png)



## [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) 随机森林



```
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier()
```

```
RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators='warn',
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)

Fitting time: 0.025 s
Predicting time: 0.001 s
score: 0.92 
accuracy: 0.92 
precision: 0.95062 
recall: 0.92771 
F1: 0.93902
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/Random%20Forest.18y60hsykn9.png)

## [AdaBoost](<https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html>)

AdaBoost，有时也叫“被提升的决策树”

```
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier()
```

```
AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.0,
                   n_estimators=50, random_state=None)

Fitting time: 0.119 s
Predicting time: 0.014 s
score: 0.924 
accuracy: 0.924 
precision: 0.94545 
recall: 0.93976 
F1: 0.9426
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/AdaBoost.8c71fz81q5i.png)



# 总结

**根据F1值排序**


|      | algorithm     | score | fitting_time | accuracy | precision | recall  | F1      | predicting_time |
| ---- | ------------- | ----- | ------------ | -------- | --------- | ------- | ------- | --------------- |
| 5    | AdaBoost      | 0.924 | 0.119        | 0.924    | 0.94545   | 0.93976 | 0.9426  | 0.014           |
| 1    | SVM           | 0.92  | 0.007        | 0.92     | 0.94512   | 0.93373 | 0.93939 | 0.001           |
| 3    | KNN           | 0.92  | 0.002        | 0.92     | 0.94512   | 0.93373 | 0.93939 | 0.019           |
| 4    | Random Forest | 0.92  | 0.025        | 0.92     | 0.95062   | 0.92771 | 0.93902 | 0.001           |
| 2    | Decision Tree | 0.908 | 0.002        | 0.908    | 0.91813   | 0.94578 | 0.93175 | 0.001           |
| 0    | Naive Bayes   | 0.884 | 0.004        | 0.884    | 0.89143   | 0.93976 | 0.91496 | 0               |


**根据训练时间排序**

|      | algorithm     | score | fitting_time | accuracy | precision | recall  | F1      | predicting_time |
| ---- | ------------- | ----- | ------------ | -------- | --------- | ------- | ------- | --------------- |
| 2    | Decision Tree | 0.908 | 0.002        | 0.908    | 0.91813   | 0.94578 | 0.93175 | 0.001           |
| 3    | KNN           | 0.92  | 0.002        | 0.92     | 0.94512   | 0.93373 | 0.93939 | 0.019           |
| 0    | Naive Bayes   | 0.884 | 0.004        | 0.884    | 0.89143   | 0.93976 | 0.91496 | 0               |
| 1    | SVM           | 0.92  | 0.007        | 0.92     | 0.94512   | 0.93373 | 0.93939 | 0.001           |
| 4    | Random        | 0.92  | 0.025        | 0.92     | 0.95062   | 0.92771 | 0.93902 | 0.001           |
| 5    | AdaBoost      | 0.924 | 0.119        | 0.924    | 0.94545   | 0.93976 | 0.9426  | 0.014           |

本项目中各个算法对比可发现，AdaBoost，SVM准确率比较高，但是训练耗时比较久。

朴素贝叶斯，决策树训练最快，但是准确率不是很高。
