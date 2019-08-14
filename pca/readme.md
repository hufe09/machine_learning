# 主成分分析 PCA

PCA (Principal Component Analysis) 是一套全面应用于各类数据分析的分析方法，这些分析包括特征集压缩。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565246449850.1texfw2delq.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565246741901.cniss35jlh7.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565247687680.injpxusawjd.png)

## 哪些数据可用于PCA



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565248206047.5vc98ptgb5f.png)

主成分分析法的部分优点在于，数据不一定是完美的1D才能找到主轴！

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565248709102.n2ua2xw3vck.png)

## 可测量的特征与潜在的特征练习

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565248864585.e488mgrg71b.png)

## 从四个特征到两个

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565249160436.t6u1m8rn4i8.png)

这是一个典型的回归练习，因为我们预期的输出是连续性的，所以使用分类器是不合适的。我们希望用回归从而得到一个数字作为输出结果，在这里这个数字就是房价。

## 哪个是最合适的选择参数的工具？

- SelectKBest（K 为要保留的特征数量）
- SelectPercentile

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565249690120.3r2s0yf0dgc.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565249993586.bzft0dl292t.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565250868236.izd3byauom.png)



## 你认为我们为什么以这种方式确定主成分的呢？

- 计算复杂度低
- 可以最大程度保留来自原始数据的信息量
- 只是一种惯例，并没有什么实际的原因

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565251586864.mmg1egfjw6.png)

## 当我们进行投射时，绿点和黄点哪个会丢失更多的信息呢？

- 绿点
- 黄点



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565251803827.jg4gu85l7to.png)



## 最大主成分数量

假设你的数据集里有100个训练点，每个点有4个特征，在 sklearn 允许的范围内，你利用其他的 PCA 实现方式所能找到的主要成分的最大数量是多少？

-  4
-  100

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565252863459.8j5af84xic8.png)

That's right.  If you look at the sklearn documentation for PCA, you'll see that the max number of PCs is min(n_features, n_data_points).  In this case, that means min(4, 100), or 4.

## PCA的回顾/定义

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565253055244.5w56xil04dq.png)

- PCA是将输入特征转化为其主要成分的系统化方式。这些主成分可供后面使用，而非原始输入特征。
- 将其用作回归或分类任务中的新特征。
- 主成分的定义是数据中会使方差最大化的方向，它可以在你对这些主成分执行投影或压缩时，将出现信息丢失的可能性降至最低。
- 对主成分划分等级，数据因特定主成分而产生的方差越大，那么该主成分的等级越高。因此，产生方差最大的主成分即为第一个主成分，产生方差第二大的则为第二个主成分，以此类推。
- 主成分某种意义上是互相垂直的，从数学角度出发，第二个主成分绝对不会与第一个主成分重叠，第三个也不会通过第二个与第一个重叠等。因此，可以将他们当做单独的特征对待。
- 主成分数量是有上限的，该最大值等于数据集中的数据集中的输入特征数量。通常情况下，只会使用前几个主成分，但也可以一直使用下去，达到最大值，但这样做不能为你带来真正的收获，而只通过不同的方式展示特征。如果你在回归中或分类任务中同时使用所有主成分，PCA不会出错，但对比仅使用初始输入特征并没有任何优势。

## 何时使用 PCA

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565256526273.rgn7uzajf7o.png)

- 如果你想访问隐藏的特征，而你认为这些特征可能显示在数据的图案中，要做的工作可能就是确定是否存在隐藏的特征。换句话讲，你只想知道第一个主成分的大小。举例，是否可以估量出 Enron 的大亨是谁。
- 第二种情况是降维，PCA 可以执行许多工作，能在该方面为你提供帮助。
  - 可以帮助可视化高维数据。当你画散点图的时候，只有两个维度可用，很多情况下会超过两个特征，如果在只有两个维度的情况下，画出能够表示数据点的三个，四个或更多的特征，能做的就是将其投射到前两个主成分，然后只要标绘并画出散点图。
  - 你怀疑数据中存在噪音的情况，几乎所有数据都存在噪音，希望第一个或第二个，也就是最强大的主成分，捕获数据中真正的模式，而较小的主成分只表示这些模式的噪音变体，因此，通过抛弃重要性较低的主成分，就可以去除这些噪音。
  - 在使用算法之前使用 PCA 进行预处理。如果有很高的维数，而且算法比较复杂，比如分类算法，则算法的方差非常高，最终会被数据中的噪音同化，导致运行非常慢。因此，我们可以执行的操作之一就是使用 PCA 降低输入特征的维数，这样分类算法可以更好地发挥作用。

 

## 为什么PCA在人脸识别中有不错的应用呢？

- 人脸照片通常有很高的输入维度（很多像素）
- 人脸具有一些一般性形态，这些形态可以以较小维数的方式捕捉，比如人一般都有两只眼睛，眼睛基本都位于接近脸的顶部的位置
- 使用机器学习技术，人脸识别是非常容易的（因为人类可以轻易做到）

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565257730656.y1hmb3ckib.png)



# Mini-project 

Scikit-learn 示例中的人脸识别项目：[使用特征脸和SVM的识别](<https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html>)

此示例中使用的数据集是“野外标记面孔”（又名[LFW](http://vis-www.cs.umass.edu/lfw/)）的预处理摘录：

> [http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz(233MB](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz) )

数据集中代表人数最多的前5位的预期结果：

|                    |      |      |      |      |
| :----------------- | :--- | :--- | :--- | :--- |
| 阿里尔沙龙         | 0.67 | 0.92 | 0.77 | 13   |
| 科林鲍威尔         | 0.75 | 0.78 | 0.76 | 60   |
| 唐纳德拉姆斯菲尔德 | 0.78 | 0.67 | 0.72 | 27   |
| 乔治W布什          | 0.86 | 0.86 | 0.86 | 146  |
| 格哈德施罗德       | 0.76 | 0.76 | 0.76 | 25   |
| 雨果查韦斯         | 0.67 | 0.67 | 0.67 | 15   |
| 托尼布莱尔         | 0.81 | 0.69 | 0.75 | 36   |
| 平均/总计          | 0.80 | 0.80 | 0.80 | 322  |

- ![sphx_glr_plot_face_recognition_001.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_face_recognition_001.png) 
- ![sphx_glr_plot_face_recognition_002.png](https://scikit-learn.org/stable/_images/sphx_glr_plot_face_recognition_002.png)

如果直接运行下载的代码，会先下载233MB的数据文件。你可以点击[这里](http://cn-static.udacity.com/mlnd/eigenfaces.zip)先下载数据集，再根据指示运行代码。

然后将下载文件包中的 `scikit_learn_data/lfw_home` 文件夹放入本机的 `scikit_learn_data`文件夹下面。

下面是查看`scikit_learn_data`的代码。

```
from sklearn.datasets.base import get_data_home 
print (get_data_home()) 
```

```
C:\Users\hufe\scikit_learn_data
```

## Let’s go!

```
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

from sklearn.decomposition import PCA

n_components = 150
# n_components = 10 # [10, 15, 25, 50, 100, 250]

print("Extracting the top %d eigenfaces from %d faces" % (n_components, X_train.shape[0]))
t0 = time()
pca = PCA(n_components=n_components, whiten=True).fit(X_train)
print(f"PCA可释方差{pca.explained_variance_ratio_}")
print(" PCA done in %0.3fs" % (time() - t0))

eigenfaces = pca.components_.reshape((n_components, h, w))
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("构建显示主要分类指标的文本报告")
print(classification_report(y_test, y_pred, target_names=target_names))
mat = confusion_matrix(y_test, y_pred, labels=list(range(n_classes)))
print(mat)
```



```
['Ariel Sharon' 'Colin Powell' 'Donald Rumsfeld' 'George W Bush'
 'Gerhard Schroeder' 'Hugo Chavez' 'Tony Blair']
Images shape: (1288, 50, 37)
Total dataset size:
n_samples: 1288
n_features: 1850
n_classes: 7
Extracting the top 150 eigenfaces from 966 faces
 PCA done in 0.344s
Projecting the input data on the eigenfaces orthonormal basis
done in 0.021s
Fitting the classifier to the training set
done in 57.055s
Best estimator found by grid search:
Predicting the people names on the testing set
done in 0.092s
构建显示主要分类指标的文本报告
                   precision    recall  f1-score   support
     Ariel Sharon       0.78      0.54      0.64        13
     Colin Powell       0.83      0.87      0.85        60
  Donald Rumsfeld       0.94      0.63      0.76        27
    George W Bush       0.82      0.98      0.89       146
Gerhard Schroeder       0.95      0.80      0.87        25
      Hugo Chavez       1.00      0.47      0.64        15
       Tony Blair       0.97      0.81      0.88        36
         accuracy                           0.85       322
        macro avg       0.90      0.73      0.79       322
     weighted avg       0.87      0.85      0.85       322
[[  7   1   0   5   0   0   0]
 [  1  52   0   7   0   0   0]
 [  1   2  17   7   0   0   0]
 [  0   3   0 143   0   0   0]
 [  0   1   0   3  20   0   1]
 [  0   3   0   4   1   7   0]
 [  0   1   1   5   0   0  29]]
```
## 绘制热力图

```
import matplotlib.pyplot as plt
import seaborn as sns
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=lfw_people.target_names,
            yticklabels=lfw_people.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.3ulguir5f3g.png)

## 每个主成分的可释方差

- 我们提到 PCA 会对主成分进行排序，第一个主成分具有最大方差，第二个主成分 具有第二大方差，依此类推。第一个主成分可以解释多少方差？第二个呢？
```
print(f"PCA可释方差{pca.explained_variance_ratio_}")
```
```
[0.19334696 0.15120733 0.07087097 0.05947997 0.05157473 0.02887115
 0.02516417 0.02175813 0.02019795 0.01902674 0.01682204 0.01581115
 0.01222641 0.01088175 0.01064345 0.00979791 0.00892528 0.00854599
 0.00835731 0.00722789 0.00696536 0.00654065 0.00639635 0.00561641
 0.00531133 0.00520193 0.00507621 0.00484172 0.0044369  0.00417892
 0.00393588 0.00382102 0.00356086 0.0035129  0.00334764 0.00330026
 0.00314765 0.00296333 0.00290273 0.00284843 0.0027972  0.00267738
 0.00259944 0.00258424 0.00240935 0.00239153 0.00235194 0.002226
 0.00217408 0.00216579 0.0020904  0.00205468 0.00200501 0.00197532
 0.00193916 0.00189068 0.00180148 0.00179006 0.00174791 0.00173165
 0.0016573  0.00163091 0.0015747  0.00153471 0.00149907 0.00147226
 0.00143931 0.00141966 0.00139656 0.00138275 0.00134076 0.00133287
 0.00128757 0.00125612 0.00124358 0.00121946 0.0012092  0.00118327
 0.00115159 0.00113808 0.00112684 0.00111697 0.00109382 0.00107228
 0.00105699 0.00104355 0.00102401 0.00101715 0.00099798 0.00096344
 0.00094243 0.00092123 0.0009141  0.00089189 0.00087181 0.00086293
 0.00084341 0.00083965 0.00082863 0.00080368 0.00078822 0.00078146
 0.00075766 0.00075353 0.00074777 0.00073438 0.00073248 0.00071662
 0.00070629 0.00069761 0.00066924 0.00066539 0.00065642 0.00064062
 0.00063728 0.00062724 0.00061843 0.00061075 0.00060227 0.00059362
 0.00058293 0.0005755  0.00056642 0.00056267 0.00055067 0.00054504
 0.00053292 0.00052335 0.00051481 0.00051407 0.0005105  0.00049529
 0.00048843 0.00047819 0.00047188 0.00046608 0.00046503 0.00045426
 0.00044859 0.00044533 0.0004386  0.00043202 0.00042959 0.00042076
 0.00041274 0.00041019 0.0004037  0.00040245 0.00039159 0.00038644]
```



## 要使用多少个主成分

现在你将尝试保留不同数量的主成分。在类似这样的多类分类问题中（要应用两个以上标签），准确性这个指标不像在两个类的情形中那么直观。相反，更常用的指标是 F1 分数。

我们将在评估指标课程中学习 F1 分数，但你自己要弄清楚好的分类器的特点是具有高 F1 分数还是低 F1 分数。你将通过改变主成分数量并观察 F1 分数如何相应地变化来确定。

- 将更多主成分添加为特征以便训练分类器时，你是希望它的性能更好还是更差？

更好。理想情况下，我们希望添加更多的组件将为我们提供更多的信号信息，以提高分类器的性能。



- 精度(precision) = 正确预测的个数(TP)/被预测正确的个数(TP+FP)
- 召回率(recall)=正确预测的个数(TP)/预测个数(TP+FN)
- F1 = 2精度召回率/(精度+召回率)

## F1分数与使用主成分数

将 `n_components` 更改为以下值：[10, 15, 25, 50, 100, 250]。对于每个主成分，请注意 Ariel Sharon 的 F1 分数。（对于 10 个主成分，代码中的绘制功能将会失效，但你应该能够看到 F1 分数。）


-  `n_components=10` 

```
                  precision    recall  f1-score   support
     Ariel Sharon       0.10      0.15      0.12        13
     Colin Powell       0.44      0.52      0.47        60
  Donald Rumsfeld       0.27      0.37      0.31        27
    George W Bush       0.67      0.58      0.62       146
Gerhard Schroeder       0.18      0.20      0.19        25
      Hugo Chavez       0.30      0.20      0.24        15
       Tony Blair       0.50      0.39      0.44        36
         accuracy                           0.47       322
        macro avg       0.35      0.34      0.34       322
     weighted avg       0.50      0.47      0.48       322
```

  - `n_components=15` 

```
                  precision    recall  f1-score   support
     Ariel Sharon       0.25      0.46      0.32        13
     Colin Powell       0.66      0.73      0.69        60
  Donald Rumsfeld       0.47      0.59      0.52        27
    George W Bush       0.84      0.67      0.75       146
Gerhard Schroeder       0.39      0.44      0.42        25
      Hugo Chavez       0.46      0.40      0.43        15
       Tony Blair       0.51      0.56      0.53        36
         accuracy                           0.62       322
        macro avg       0.51      0.55      0.52       322
     weighted avg       0.66      0.62      0.64       322
```

  - `n_components=25` 

```
                   precision    recall  f1-score   support
     Ariel Sharon       0.56      0.69      0.62        13
     Colin Powell       0.72      0.87      0.79        60
  Donald Rumsfeld       0.48      0.52      0.50        27
    George W Bush       0.86      0.82      0.84       146
Gerhard Schroeder       0.58      0.56      0.57        25
      Hugo Chavez       0.82      0.60      0.69        15
       Tony Blair       0.69      0.61      0.65        36
         accuracy                           0.74       322
        macro avg       0.67      0.67      0.67       322
     weighted avg       0.75      0.74      0.74       322
```

  - `n_components=50` 

```
                   precision    recall  f1-score   support
     Ariel Sharon       0.62      0.77      0.69        13
     Colin Powell       0.83      0.92      0.87        60
  Donald Rumsfeld       0.70      0.59      0.64        27
    George W Bush       0.87      0.90      0.89       146
Gerhard Schroeder       0.75      0.72      0.73        25
      Hugo Chavez       0.77      0.67      0.71        15
       Tony Blair       0.86      0.69      0.77        36
         accuracy                           0.83       322
        macro avg       0.77      0.75      0.76       322
     weighted avg       0.83      0.83      0.82       322
```

  - `n_components=100` 

```
                   precision    recall  f1-score   support
     Ariel Sharon       0.69      0.69      0.69        13
     Colin Powell       0.79      0.88      0.83        60
  Donald Rumsfeld       0.82      0.67      0.73        27
    George W Bush       0.88      0.94      0.91       146
Gerhard Schroeder       0.87      0.80      0.83        25
      Hugo Chavez       0.90      0.60      0.72        15
       Tony Blair       0.91      0.81      0.85        36
         accuracy                           0.85       322
        macro avg       0.84      0.77      0.80       322
     weighted avg       0.86      0.85      0.85       322
```
  - `n_components=250` 

```
                  precision    recall  f1-score   support
     Ariel Sharon       0.60      0.69      0.64        13
     Colin Powell       0.74      0.90      0.81        60
  Donald Rumsfeld       0.82      0.67      0.73        27
    George W Bush       0.90      0.90      0.90       146
Gerhard Schroeder       0.87      0.80      0.83        25
      Hugo Chavez       0.80      0.53      0.64        15
       Tony Blair       0.82      0.75      0.78        36
         accuracy                           0.83       322
        macro avg       0.79      0.75      0.76       322
     weighted avg       0.84      0.83      0.83       322
```
**F1-score汇总对比：**

| n_components      | 10   | 15   | 25   | 50   | 100  | 150  | 250  |
| ----------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| Ariel Sharon      | 0.12 | 0.32 | 0.62 | 0.69 | 0.69 | 0.64 | 0.64 |
| Colin Powell      | 0.47 | 0.69 | 0.79 | 0.87 | 0.83 | 0.85 | 0.81 |
| Donald Rumsfeld   | 0.31 | 0.52 | 0.5  | 0.64 | 0.73 | 0.76 | 0.73 |
| George W Bush     | 0.62 | 0.75 | 0.84 | 0.89 | 0.91 | 0.89 | 0.9  |
| Gerhard Schroeder | 0.19 | 0.42 | 0.57 | 0.73 | 0.83 | 0.87 | 0.83 |
| Hugo Chavez       | 0.24 | 0.43 | 0.69 | 0.71 | 0.72 | 0.64 | 0.64 |
| Tony Blair        | 0.44 | 0.53 | 0.65 | 0.77 | 0.85 | 0.88 | 0.78 |
| accuracy          | 0.47 | 0.62 | 0.74 | 0.83 | 0.85 | 0.85 | 0.83 |
| macro avg         | 0.34 | 0.52 | 0.67 | 0.76 | 0.8  | 0.79 | 0.76 |
| weighted avg      | 0.48 | 0.64 | 0.74 | 0.82 | 0.85 | 0.85 | 0.83 |

**如果看到较高的 F1 分数，这意味着分类器的表现是更好还是更差？**

更好。

## 维度降低与过拟合

**在使用大量主成分时，是否看到过拟合的任何证据？**PCA 维度降低是否有助于提高性能？

当使用大量 PC 时，你是否会看到任何过拟合迹象？

- 不会，PC 越多性能通常越好 

- 不会，更多 PC 不会导致性能变化 

- 需要更多信息 

- 会，PC 较多时性能会下降（**select**）



## 选择主成分

考虑你应该选择多少主成分。 对于你应该使用多少主成分没有简单明了的答案，你必须弄清楚。 

计算出要使用多少主成分的好方法是什么？

- 一般情况下，总选择靠前的10% 

- 针对不同主成分数量进行训练，然后观察，对于每一个可能的主要成分数量，算法的查准率如何，重复几次后发现，在某一点上会出现收益递减的情况，也就是添加更多的主成分之后，你的结果没有太大差异。如果数据增长停滞时，你就应该停止增加主要成分了。（select）

- 在将特征输入PCA之前，对其进行特征选择，在完成选择之后，你就可以使用输入特征中的所有主要成分。

主成分分析将会找到一种方法，将来自许多不同输入特征的信息合并起来，所以如果在进行 PCA 之前排除所有输入特征，那么在某种意义上，你也就排除了 PCA 能够挽救的信息。在制定主成分之后对其进行特征选择是可以的，但是如果在执行 PCA 之前排除信息，需要特别小心。 PCA 的计算量可能非常大，所以如果你有非常大的输入特征空间，并且你知道其中许多是完全不相关的特征。尽管予以舍弃，但是要小心行事。