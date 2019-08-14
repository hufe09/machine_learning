# SVM 支持向量机

SVM(Support Vector Machine),常见的一种分类方法，在机器学习中，SVM 是有监督的学习模型。

简单来讲，SVM 就是帮我们找到两类数据之间的超平面的分隔线，或者通常称为超平面。用 SVM 计算的过程就是帮我们找到那个超平面的过程，这个超平面就是 **SVM 分类器**。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.t2u0i7hnspq.png)



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.pyv6j4rnkrc.png)

这条线使距离最近点的距离最大化，相对于分类来说，这条线最大化了与左右两分类最近点的距离，这种距离成为**间隔**。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.uchlry2kkj.png)

## SVM 对异常值的响应

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ljphil6y2bh.png)

- 有时候似乎支持向量机无法对某些问题进行正确分类，比如存在异常值，我们希望SVM尽可能得到一个最好的结果。

## 非线性数据

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.gal7c10u4ur.png)

## 新特征

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.zr7djuqqqu.png)

## 可视化新特征

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.obuzusybzs.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.6wrie0zajen.png)

## 核技巧

**核函数。它可以将样本从原始空间映射到一个更高维的特质空间中，使得样本在新的空间中线性可分**。

核函数接收低纬度的输入空间或特征空间，并将其映射到高纬度空间，所以，过去不可线性分割的内容变为可分割。

此为支持向量机的一个优点，可以便捷地找到最佳的线性分类器，或者不同的线性分隔线，在高维度空间应用所谓的核技巧。

也就是实现非线性分隔线对数据集进行分类。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.yjy3p8xujr.png)

## kernel 和 gamma

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.wem35fi1g7.png)

## SVM C参数

C会在光滑的决策边界，以及尽可能正确分类所有训练点两者之间进行平衡。

**C 的作用是什么？一个更大的 C 会让边界更平滑还是得到更多正确的训练点？**

- □ 更平滑
- □ 更多正确的训练点

>更平滑

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.1y1e7z3tkmv.png)

## 过拟合

可能影响过拟合的参数?

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.mebpaiaqp5.png)



### 如何创建一个 SVM 分类器呢？

我们首先使用 SVC 的构造函数：`model = svm.SVC(kernel=‘rbf’, C=1.0, gamma=‘auto’)`，这里有三个重要的参数 `kernel`、`C` 和 `gamma`。

- `kernel` 代表核函数的选择，它有四种选择，只不过默认是 `rbf` ，即高斯核函数。
    1. linear：线性核函数
    2. poly：多项式核函数
    3. rbf：高斯核函数（默认）
    4. sigmoid：sigmoid 核函数    
    
    这四种函数代表不同的映射方式，你可能会问，在实际工作中，如何选择这 4 种核函数呢？下面解释一下：
  
      - a. 线性核函数，是在数据线性可分的情况下使用的，运算速度快，效果好。不足在于它不能处理线性不可分的数据。
    
      - b. 多项式核函数可以将数据从低维空间映射到高维空间，但参数比较多，计算量大。
    
      - c. 高斯核函数同样可以将样本映射到高维空间，但相比于多项式核函数来说所需的参数比较少，通常性能不错，所以是默认使用的核函数。
    
      - d. sigmoid 核函数，SVM 实现的是多层神经网络。如果了解深度学习，应该知道 sigmoid 经常用在神经网络的映射中。
  
    上面介绍的 4 种核函数，除了第一种线性核函数外，其余 3 种都可以处理线性不可分的数据。

- 参数 `C` 代表目标函数的惩罚系数，惩罚系数指的是分错样本时的惩罚程度，默认情况下为 1.0。当 C 越大的时候，分类器的准确性越高，但同样容错率会越低，泛化能力会变差。相反，C 越小，泛化能力越强，但是准确性会降低。

- 参数 `gamma` 代表核函数的系数，默认为样本特征数的倒数，即 $gamma = 1 / n\_features$。



# Mini-project

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

### 核函数选择：高斯函数

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

## 部署 SVM 最后提醒

希望 Sebastian 在说朴素贝叶斯非常适合文本时，更清楚地表达了他的意思。对于这一具体问题，朴素贝叶斯不仅更快，而且通常比 SVM 更出色。当然，SVM 更适合许多其他问题。你在第一次求解问题时就知道该尝试哪个算法，这是机器学习艺术和科学性的一个体现。除了选择算法外，视你尝试的算法而定，你还需要考虑相应的参数调整以及过拟合的可能性（特别是在你没有大量训练数据的情况下）。

我们通常建议你尝试一些不同的算法来求解每个问题。调整参数的工作量很大，但你现在只需要听完这堂课，我们将向你介绍 GridCV，一种几乎能自动查找最优参数调整的优秀 sklearn 工具。

