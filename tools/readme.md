# Mini-project

我们有一组邮件，分别由同一家公司的两个人Sara 和Chris 各自撰写其中半数的邮件。我们的目标是仅根据邮件正文区分每个人写的邮件。

我们会先给你一个字符串列表。每个字符串代表一封经过预处理的邮件的正文；然后，我们会提供代码，用来将数据集分解为训练集和测试集。

然后使用机器学习算法，根据作者对电子邮件作者 ID进行分类。

- Sara has label 0
-  Chris has label 1

## [Naive Bayes](<https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes>)

*naive_bayes/nb_author_id.py*

```
from sklearn.naive_bayes import GaussianNB
```



```
t0 = time()
clf = GaussianNB()
clf_model = clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t1, 3), "s")

print("准确率:", clf.score(features_test, labels_test))
print(clf_model)
print(pred)
```

Out:

```
no. of Chris training emails: 7936
no. of Sara training emails: 7884
training time: 2.589 s
predicting time: 0.38 s
准确率 0.9732650739476678
GaussianNB(priors=None, var_smoothing=1e-09)
[0 0 1 ... 1 0 0]
```

## [SVM](<https://scikit-learn.org/stable/modules/svm.html#svm>)

*svm/svm_author_id.py*

### 核函数选择：线性函数

```
from sklearn import svm
```

 `clf = svm.SVC(kernel="linear")`

```
t0 = time()
clf = svm.SVC(kernel="linear")
print(clf.fit(features_train, labels_train))

print("fitting time:", round(time()-t0, 3), "s")
clf.predict(features_test)
t1 = time()
print("准确率:", clf.score(features_test, labels_test))
print("predicting time:", round(time()-t1, 3), "s")

# 统计预测值次数
print(Counter(pred).items())
```
下面是结果：

```
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
fitting time: 213.111 s
准确率:  0.9840728100113766
predicting time: 22.41 s
dict_items([(0, 877), (1, 881)])
```


### 核函数选择：高斯函数

 `clf = svm.SVC(kernel="rbf", gamma="auto", C=10000)`

分类器要花很长的时间来训练


Out:
```
SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
fitting time: 151.203 s
准确率:  0.9908987485779295
predicting time: 14.036 s
```



可以看出，SVM准确率非常高，但是花费时间成本太大。

### 减小数据集

加快算法速度的一种方式是在一个较小的训练数据集上训练它。

```
features_train = features_train[:int(len(features_train)/100)]
labels_train = labels_train[:int(len(labels_train)/100)]
```

这两行有效地将训练数据集切割至原始大小的 1%，丢弃掉 99% 的训练数据。  

再次查看结果：

```
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
    kernel='linear', max_iter=-1, probability=False, random_state=None,
    shrinking=True, tol=0.001, verbose=False)
fitting time: 0.125 s
[0 1 1 ... 1 0 1]
准确率:  0.8845278725824801
predicting time: 1.276 s
```

减小数据集大小后，牺牲一些准确率可加快模型训练/预测速度。

- **非常快速地运行的算法尤其重要**
- 预测电子邮件的作者 标记信用卡诈骗，并在诈骗交易完成前进行拦截 
  
- 声音识别，如 Siri

###  核函数选择：高斯函数

`clf = svm.SVC(kernel="rbf", gamma="auto")`

```
SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
fitting time: 0.133 s
[0 1 1 ... 1 1 1]
准确率:  0.6160409556313993
predicting time: 1.479 s
```

### 改变C值

- C=1.0
`clf = svm.SVC(kernel="rbf", gamma="auto", C=1.0)`

Out:
```
fitting time: 0.135 s
准确率:  0.6160409556313993
predicting time: 1.454 s
dict_items([(0, 218), (1, 1540)])
```

- C=100
`clf = svm.SVC(kernel="rbf", gamma="auto", C=100)`

Out:
```
fitting time: 0.151 s
准确率:  0.6160409556313993
predicting time: 1.42 s
dict_items([(0, 218), (1, 1540)])
```

- C=1000
`clf = svm.SVC(kernel="rbf", gamma="auto", C=1000)`

Out:
```
fitting time: 0.128 s
准确率:  0.8213879408418657
predicting time: 1.798 s
dict_items([(0, 581), (1, 1177)])
```

- C=10000
`clf = svm.SVC(kernel="rbf", gamma="auto", C=10000)`

Out:
```
fitting time: 0.127 s
准确率:  0.8924914675767918
predicting time: 1.125 s
dict_items([(0, 740), (1, 1018)])
```



### 优化后的 RBF 与线性 SVM
- 问题：
    - 准确率
    - SVM（0 或 1，分别对应 Sara 和 Chris）为测试集的元素 10 预测的类是多少？26？50？
    - 有多少预测属于“Chris”(1) 类？

```
t0 = time()
clf = svm.SVC(kernel="rbf", gamma="auto", C=10000)
print(clf.fit(features_train, labels_train))

print("fitting time:", round(time()-t0, 3), "s")
pred = clf.predict(features_test)

t1 = time()
print('准确率: ', clf.score(features_test, labels_test))
print("predicting time:", round(time()-t1, 3), "s")

# 统计预测值次数
print(Counter(pred).items())

# 预测某元素的值
print(f'测试集的元素为10:{pred[10]}, 测试集的元素为26:{pred[26]}, 测试集的元素为50:{pred[50]}')
```

Out:
```
SVC(C=10000, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
fitting time: 151.203 s
准确率:  0.9908987485779295
predicting time: 14.036 s
dict_items([(0, 881), (1, 877)])
测试集的元素为10:1, 测试集的元素为26:0, 测试集的元素为50:1
```

朴素贝叶斯非常适合文本，对于这一具体问题，朴素贝叶斯不仅更快，而且通常比 SVM 更出色。当然，SVM 更适合许多其他问题。

## [Decision Trees](<https://scikit-learn.org/stable/modules/tree.html#tree>)

*decision_tree/dt_author_id.py*

```
from sklearn import tree
```

```
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
```

Out:
```
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                       max_features=None, max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=40,
                       min_weight_fraction_leaf=0.0, presort=False,
                       random_state=None, splitter='best')
fitting time: 102.382 s
准确率:  0.9766780432309442
predicting time: 0.053 s
dict_items([(0, 896), (1, 862)])
```



数据中的特征数是多少？

```
len(features_train[0])
Out[3]: 3785
```

当你仅使用 1% 的可用特征（即百分位数 = 1）时

```
len(features_train[0])
Out[3]: 379
```

决策树的准确率是多少？

```
fitting time: 6.405 s
准确率:  0.9664391353811149
predicting time: 0.005 s
dict_items([(0, 872), (1, 886)]
```

**在其他所有方面都相等的情况下，特征数量越多会使决策树的复杂性更高还是更低？**

特征越少，意味着决策树在找到决策面时划出非常具体的小点的机会就越少。这些特定的小点(我们也称之为高方差结果的证据)表明了一个更复杂的决策过程。所以拥有更多的特性并不意味着你有一个不那么复杂的决策树。

