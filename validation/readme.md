

# 交叉验证 Cross validation

## 何处使用训练与测试数据

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.3bsqr8fu5zr.png)





GridSearchCV 用于系统地遍历多种参数组合，通过交叉验证确定最佳效果参数。它的好处是，只需增加几行代码，就能遍历多种组合。

下面是来自 [sklearn 文档](<https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>)的一个示例：

```
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
svc = svm.SVC()
clf = grid_search.GridSearchCV(svc, parameters)
clf.fit(iris.data, iris.target)
```

让我们逐行进行说明。

```
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
```

参数字典以及他们可取的值。在这种情况下，他们在尝试找到 kernel（可能的选择为 'linear' 和 'rbf' ）和 C（可能的选择为1和10）的最佳组合。

这时，会自动生成一个不同（kernel、C）参数值组成的“网格”:

| ('rbf', 1)    | ('rbf', 10)    |
| :------------ | :------------- |
| ('linear', 1) | ('linear', 10) |

各组合均用于训练 SVM，并使用交叉验证对表现进行评估。

`svc = svm.SVC()` 
这与创建分类器有点类似，就如我们从第一节课一直在做的一样。但是请注意，“clf” 到下一行才会生成—这儿仅仅是在说采用哪种算法。另一种思考方法是，“分类器”在这种情况下不仅仅是一个算法，而是算法加参数值。请注意，这里不需对 kernel 或 C 做各种尝试；下一行才处理这个问题。

`clf = grid_search.GridSearchCV(svc, parameters)` 
这是第一个不可思议之处，分类器创建好了。 我们传达算法 (svc) 和参数 (parameters) 字典来尝试，它生成一个网格的参数组合进行尝试。

`clf.fit(iris.data, iris.target)` 
第二个不可思议之处。 拟合函数现在尝试了所有的参数组合，并返回一个合适的分类器，自动调整至最佳参数组合。现在您便可通过 `clf.best_params_` 来获得参数值。

## K折交叉验证 K-fold Cross validation

对收集的数据创建模型 可以有许多方式，无论是上节课中创建虚拟变量，或高阶项，还是填充缺失值的其他技巧之一

要测量数据，**Scikit-learn** 和 **Pandas** 是两个著名的特征工程库，在 scikit-learn 预处理中， 有很多方法，例如：

- StandardScaler
- MinMax scaler MaxAbs Scaler 最小最大缩放器
- QuantileTransformer 量化转换器
- 归一化
- 二进制化
- 编码类属特征
- 输入缺失值
- 自定义转换



对于模型评估的测量标准，不同方法存在巨大争议，常见测量标准包括 **R 平方和均方差**，其他测量方法包括赤池信息准则 (AIC)，贝叶斯信息准则 (BIC) 和马洛斯的 CP 值，还有许多其他我没有提过的测量标准，可以评价你的回归模型，或其他模型是否较好地拟合数据，这些测量标准的值取决于数据的缩放比例，和预测反应变量的变量，根据我的经验 人们在实际中倾向于使用 R 平方和均方差，但是实际上这些方法在模型拟合中存在误导性，如果我们只验证对同一个数据集的做法，事实证明 无论我们向模型中添加什么变量，都会提升预测，这就是 R 平方值增加和均方差下降的原因

那么我们怎样知道增加变量，是否真的改善了模型与数据的拟合度呢？这里有个非常有效的技巧 **称为交叉验证**，能够恰好预测这个想法，即增加越多变量 可以认为模型总体得到改善

在接下来的概念中，你将学习交叉验证的原理，以及如何使用 Scikit-Learn 构建应用到实践

## 交叉验证的原理

在上一个视频中，我讨论了模型中简单地添加越来越多的变量，可以产生均方差值中更优的 R 平方，已经介绍过交叉验证这种观点

在交叉验证中我们，对数据子集训练回归模型 **称为训练数据**，我们测试模型如何较好地运行 **称为测试集**，在交叉验证中我们通过多次完成这些部分，确保模型也可以通用化，Sebastian 将会向你一步步展示如何做到，然后我们在 Python 中进行练习

![image](https://camo.githubusercontent.com/1ffc780096a5da647ff97b9901add4a8ceba278d/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6875666530392f4769744e6f74652d496d616765732f6d61737465722f50696365652f696d6167652e68356e3762336c776c6e2e706e67)



在进入交叉验证之前 我们先来讨论一下，将数据集分拆为训练和测试数据的问题，假设这就是你的数据，现在你要说出哪个部分的数据是测试 哪个是训练，你进入的困境是希望将两个集合都做到最大化

![image](https://camo.githubusercontent.com/01084e1f212f9de8e3e4de5a15cb3bc0909e4a3e/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6875666530392f4769744e6f74652d496d616765732f6d61737465722f50696365652f696d6167652e69697a68653633396430682e706e67)

你希望在训练集中有尽可能多的数据点，以获得最佳学习结果 同时也希望测试集中有最大数量的数据项，来获得最佳验证，但显然这里需要进行折衷 每当你从训练集中取出一个数据点拿去测试，训练集中就会少一个数据点，所以 我们要重新设定这个折衷，这就是涉及到交叉验证的地方，基本要点是将训练数据平分到相同大小的 k 个容器内

例如有 200 个训练数据点，有十个容器，很快便可得出，每个容器内有多少个数据点呢？很明显 是 20，所以 10 个容器内分别有 20 个数据点，然而在 Katie 讲述的操作中 你只是挑选其中一个容器，作为验证容器 另一个作为训练容器，在 k 折交叉验证中 你将运行 k 次单独的学习试验，在每次试验中 **你将从这 k 个子集中挑选一个作为验证集，剩下 k-1 个容器放在一起作为训练集**，然后训练你的机器学习算法，与以前一样 将在验证集上验证性能，交叉验证中的要点是这个操作会运行多次，在此例中为十次 然后将十个不同的测试集，对于十个不同的保留集的表现进行平均，就是将这 k 次试验的测试结果取平均值，显然 这会花更多的计算时间 因为你要运行，k 次单独的学习试验，但学习算法的评估将更加准确，从某种程度上讲 你差不多使用了全部训练数据进行训练，以及全部训练数据进行验证

![image](https://camo.githubusercontent.com/1e901379c2289651ea67ef93a388be29a5c2fcca/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6875666530392f4769744e6f74652d496d616765732f6d61737465722f50696365652f696d6167652e6764393064697165736e622e706e67)

比如我们提一个问题，假设你可以选择按照 Katie 讲述的静态训练测试方法来操作，也可以按照 10 折交叉验证来操作，你实际关心的是将训练时间降至最低，使用机器学习算法进行训练后将运行时间降至最低，忽略训练时间 将查准率提至最高，在这三种情形下 你可以挑选训练/测试或者，10 折交叉验证，告诉我你的最佳猜测，你会选择哪一种？对于每个最短的训练时间，请在右边选择两者之一

![image](https://camo.githubusercontent.com/06c2f90aff9dbd65ec82181015f84eee82c9a7d1/68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f6875666530392f4769744e6f74652d496d616765732f6d61737465722f50696365652f696d6167652e6c35626b6f75756574392e706e67)

1. 一次训练会快于10次训练，那么很显然训练集/测试集拆分很可能是更好的选择，
2. 如果你想最小化运行时间，在每一个情形下你最终都得选用一个机器运行算法，那么我可以说如果你想最小化运行时间，你同样可以得到十折交叉验证的好处，从而可更好评估事态，但是，它不是很清晰 或者不是很正确，
3. 但是如果你想最大化你所评估的精确性，你的算法好到什么程度，你肯定要选择十折交叉验证，花更多点时间 你就会得到更好的结果

## Sklearn 中的 GridSearchCV

请参考[此处](http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html)的特征脸方法代码。**使用 GridSearchCV 调整了 SVM 的哪些参数？**

```
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'),
                   param_grid, cv=5, iid=False)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found by grid search:")
print(clf.best_estimator_)
```

- SVM 的哪些参数使用特征脸示例中的 GridSearchCV 进行自动调谐？

  That's right!  5 values of `C` and 6 values of `gamma` are tested out.

# Mini-project

## 第一个（过拟合）POI识别符

你将先开始构建想象得到的最简单（未经过验证的）POI 识别符。 本节课的初始代码 (*validation/validate_poi.py*) 相当直白——它的作用就是读入数据，并将数据格式化为标签和特征的列表。 创建决策树分类器（仅使用默认参数），在所有数据（你将在下一部分中修复这个问题！）上训练它，并打印出准确率。 这是一颗过拟合树，不要相信这个数字！尽管如此，准确率是多少？

从 Python 3.3 开始，字典键被处理的顺序发生了变化，顺序在每次代码运行时都会得到随机化处理。 这会造成项目代码（均在 Python 2.7 下运行）的一些兼容性问题。 要更正这个问题，向 `validate_poi.py` 调用的 `featureFormat` 添加以下参数：`sort_keys='../tools/python2_lesson13_keys.pkl`

```
data = featureFormat(data_dict, features_list, sort_keys='../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)
```

```
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(features, labels)
print(clf.score(features, labels))
```
```
0.9894
```

精确度很高，是吧？在另一个案例中，对培训数据的测试会让你觉得你做得非常好，但是正如你已经知道的，这正是坚持测试数据的目的...

## 部署训练/测试机制

[sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)

现在，你将添加训练和测试，以便获得一个可靠的准确率数字。 使用 *sklearn.model_selection* 中的 *train_test_split* 验证； 将 30% 的数据用于测试，并设置 random_state 参数为 42（random_state 控制哪些点进入训练集，哪些点用于测试；将其设置为 42 意味着我们确切地知道哪些事件在哪个集中； 并且可以检查你得到的结果）。更新后的准确率是多少？

```
0.7241379310344828
```



## [Pipeline](<https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html>) 

使用最终估算器进行变换的流水线。按顺序应用变换列表和最终估算器。

```
sklearn.pipeline.Pipeline(steps, memory=None, verbose=False)
```

管道的目的是组合几个步骤，这些步骤可以在设置不同参数的同时进行交叉验证。

*validation/pipline_learn.py*

```
pipe = Pipeline([
    ('preprocess', MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('classify', svm.SVC())
])

N_FEATURES_OPTIONS = [2, 4, 8]
C_OPTIONS = [1, 10, 100, 1000]
param_grid = [
    {
        'reduce_dim': [PCA(iterated_power=7), NMF()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
    {
        'reduce_dim': [SelectKBest(chi2)],
        'reduce_dim__k': N_FEATURES_OPTIONS,
        'classify__C': C_OPTIONS
    },
]

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)
```

> 'classify'(自己设定的名称)通过 "__" 连接 n_components 的参数（n_components），N_FEATURES_OPTIONS 代表取值范围。
> 例如，C 为支持向量机里面的一个参数设置，`svm.SVC(C= ?)`

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.p2u232gh43g.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.aw3kizwyrct.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.7ugjkcfroys.png)

执行 `grid.fit()` 函数时

（1）先执行特征预处理 `StandardScaler()` 的 `fit`  和  `transform` 函数，将执行后的结果传递给下一个参数，即 PCA

（2）上一步的数据继续执行特征降维函数` PCA()` 或 `SelectKBest()`的 `fit`  和 `transform`  函数，生成结果传递给下一步，即SVC

（3）最后执行 `scm.SVC()`.

![img](https://img-blog.csdn.net/20180411102536324?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0FteV9tbQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)



> <https://blog.csdn.net/wong2016/article/details/82810332>
>
> <https://blog.csdn.net/Amy_mm/article/details/79890979>