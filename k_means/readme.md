# K-Means K-均值

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.j7ya4b7y4o8.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.rj4bmox1fy.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.9u62xwtp78.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.2cuvr8eabdq.png)

在K-均值算法中，首先随意画出聚类中心，假设最先的猜测是两个绿色的*，显然不是正确的聚类中心，接下来还有步骤。K-均值算法分两个步骤：第一步是**分配**，第二步是**优化**。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.gezqh3miacd.png)

这些连接数据点与集群中心的蓝色线条我们把它们视为橡皮筋，这些橡皮筋的长度越短越好，在优化步骤，我们移动绿色的集群中心点，直至橡皮筋的总长度达到最小值。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.cj5l2ckp7f.png)

## K-均值的局限性

任何固定训练集的结果是否总是一样？也就是说对于一个固定的数据集，一个固定数量的簇中心在运行K-均值算法时，是否总是会得到相同的结果？

- 不会

K-均值是所谓的爬山算法，因此，它非常依赖于初始聚类中心所处位置。

>注：“簇”指 cluster center，即聚类中心。

## 反直觉的聚类

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.j1g584r6e9a.png)

你是否会认为，左边这里所有的点有可能同属一个聚类，左右两边属于两个聚类，这也称为聚类的局部最小值（Local Minimum），它取决于聚类中心点最初的设定。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.dgdvt3nn3fb.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.vqarkmvf40h.png)



## 将 k-均值聚类应用于安然财务数据

首先你将基于电子邮件 + 财务 (E+F) 数据集的两个财务特征开始执行 K-means，请查看代码并确定代码使用哪些特征进行聚类。

> [K-Means 可视化](<https://www.naftaliharris.com/blog/visualizing-k-means-clustering/>)
>[Sklearn K-means](<https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>)

# Mini-project

在此项目中，我们会将 k-均值聚类应用于安然财务数据。当然，我们最终的目标是识别相关人员；既然我们有了已标记数据，调用 k-均值聚类这种非监督式方法就不成问题。

## 聚类特征

*k_means/k_means_cluster.py*

```
class sklearn.cluster.KMeans（n_clusters = 8，init ='k-means ++'，n_init = 10，max_iter = 300，tol = 0.0001，precompute_distances ='auto'，verbose = 0，random_state = None，copy_x = True，n_jobs = None，algorithm = 'auto' ）
```

```
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(finance_features)
pred = kmeans.predict(finance_features)
```
![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/features_scatter.4xz9xnt4iy6.png)

## 部署聚类

在 financial_features 数据上部署 k-均值聚类，并将 2 个聚类指定为参数。
聚类参数：绘制 股票期权-薪资 `exercised_stock_options`-`salary`

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/clusters-2features.nsrftptj0jd.png)

## 使用3个特征聚类

向特征列表（features_list）中添加第三个特征：`total_payments`。现在使用 3 个，而不是 2 个输入特征重新运行聚类（很明显，我们仍然可以只显示原来的 2 个维度）。将聚类绘图与使用 2 个输入特征获取的绘图进行比较。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/clusters-3features.o03upjw9lln.png)

## 当你加入一些新的特征时，有测试点移动到不同的聚类中吗？

- □ 没有，所有聚类都没变
- □ 是的，有4个测试点的聚类变了
- □ 是的，有7个测试点的聚类变了
- □ 是的，有很多测试点的聚类变了

> 对比两幅散点图，有4个测试点的聚类发生变化。

# Features Scaling 特征缩放 

后面会讨论特征缩放，它是一种特征预处理，应在执行某些分类和回归任务之前执行。这里只是快速预览，概述特征缩放的功能。

- 本例中使用的 `exercised_stock_options` 特征取的最大值和最小值是什么？
- 本例中使用的 `salary` 取的最大值和最小值是什么？

> Maximum exercised_stock_options:  34348384.0
> Minimum exercised_stock_options:  3285.0
> Maximum salary:  1111258.0
> Minimum salary:  477.0



下图为执行聚类之前应用了特征缩放，将特征范围现更改为 [0.0, 1.0]，将此绘图与迷你项目开始时获取的绘图（对只两个特征进行聚类）对比，哪些数据点改变了聚类？

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.qym8xbc0vap.png)

有2个点改变了聚类。



## 哪些算法会受特征缩放影响

- 不会受特征缩放影响的算法：线性回归、决策树
  - **决策树**：会呈现出一系列的水平线和垂线，不存在两者的交换，只是在不同的方向上进行切割，在处理某一维度时不需要考虑另一个维度的情况。例如方框缩小一半即特征缩放，图中线条的位置会有所变化，但是分割顺序是一样的.它是按比例分割的，所以两个变量之间不存在交换。
  - **线性回归**：每个特征都有一个相应系数，这个系数总是与相应的特征同时出现，特征A的变化不会影响到特征B的系数（另外如果变量扩大一倍，那它的特征会缩小1/2,其结果不变），所以分割方式相同。

- 会受到影响的算法：使用RBF核函数的 SVM、K-means聚类

  - SVM 和 K-means 聚类在计算距离时，是在利用一个维度与另一个维度进行交换，例如把某一点增加一倍，那么它的值也会扩大一倍。

    
## 特征缩放 Mini-project

对你在 k-均值聚类代码的 `salary` 和 `exercised_stock_options` 特征（仅这两项特征）运用特征缩放。 原始值为 20 万美元的 `salary` 特征和原始值为 1 百万美元的 `exercised_stock_options` 特征的重缩放值会是多少？ 

*k_means/k_means_cluster.py*

```
# 特征缩放
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
min_max_x = min_max_scaler.fit_transform(finance_features)

print(f"每个功能相对缩放数据 {min_max_scaler.scale_}")

print(f"[[200000.0, 1000000.0, 1061827.0]] 特征缩放值：{min_max_scaler.transform([[200000.0, 1000000.0, 1061827.0]])}")

print(f"20万的 `salary` 特征缩放值：{200000.0 * min_max_scaler.scale_[0]}")

print(f"100万的 `exercised_stock_options` 特征缩放值：：{1000000.0 * min_max_scaler.scale_[1]}")
```

```
每个功能相对缩放数据 [9.42567478e-07 3.25033452e-08 5.79625133e-08]

[[200000.0, 1000000.0, 1061827.0]] 
特征缩放值：[[0.1885135  0.03250335 0.06154616]]

20万的 `salary` 特征缩放值：0.1885134956811558

100万的 `exercised_stock_options` 特征缩放值：：0.03250334524429255
```

## 何时部署特征缩放

有人可能会质疑是否必须重缩放财务数据，也许我们希望 10 万美元的工资和 4 千万美元的股票期权之间存在巨大差异。如果我们想基于 `from_messages`（从一个特定的邮箱帐号发出的电子邮件数）和 `salary` 来进行集群化会怎样？ 在这种情形下，特征缩放是不必要的，还是重要的？

**如果你使用 `from_messages` 和 `salary` 作为聚类的特征，尺度将是不必要的还是重要的？ **

- 无疑，它是重要的。电子邮件的数量通常在数百或数千以下，工资通常至少高出1000倍。