 

# Naive Bayes 朴素贝叶斯

> “如果一个程序在使用既有的经验（E）执行某类任务（T）的过程中被认为是 “具备学习能力的”，那么它一定需要展现出:利用现有的经验（E），不断改善其完成既定任务（T）的性能（P）的特性。”                                                            ——Tom Mitchell, 1997

**机器学习 Machine Learning** 通常分为 **监督 supervised** 和 **非监督 unsupervised** 学习，而你将在本课学到的回归则是监督机器学习的范例之一。

在监督机器学习中，你要做的是预测数据标签。一般你可能会想预测交易是否欺诈、愿意购买产品的顾客或某一地区的房价。

在非监督机器学习中，你要做的是收集同一类尚**无标签**的数据。

## 下面哪些问题可以通过监督分类解决？

- 拿一册带标签的照片，试着认出照片中的某个人
- 对银行数据进行分析，寻找异常的交易，将其标记为涉嫌欺诈
- 根据某人的音乐喜好以及其所爱音乐的特点，比如节奏或流派推荐一首他们可能会喜欢的歌
- 根据优达学城学生的学习风格，将其分成不同的组群

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.8v1y6aoblph.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.2c7ks7jsk87.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.sd5qoq71bte.png)

## 良好的据决策面

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.cjwkqoshu2.png)



# 贝叶斯规则 Bayes Ruler


问的问题是：1％的人口患有癌症。鉴于如果您患有癌症，您有90％的可能性检测阳性，如果您没有癌症，您有90％的机会检测为阴性，如果您检测出阳性，您患癌症的可能性是多少？
![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ldowhgvgmlq.png)

## 贝叶斯规则图：

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.j5jctjmu10t.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.g5b7qbtvrwd.png)

## 可视化贝叶斯定理

在我们深入研究公式之前，让我们使用维恩图快速可视化贝叶斯定理。

想象一下以下场景：

- 我们的宇宙中有100个人
- 100人中有5人患有疾病
- 该疾病的检测准确率为90％
- 14人测试阳性

这个人患有这种疾病的可能性是多少，因为他们检测出这种疾病是正面的？

直觉上，我们可能会说有90％的可能性患有这种疾病。但事实并非如此！我们可以用这些维恩图来形象化原因：

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.u6ty36i3pu.png)
[图片来源](https://lavanya.ai/2019/05/16/bayes-theorem/)

90％的人患有该病的阳性结果（绿色圆圈）。因此，有疾病的人（蓝色圆圈）的90％与他们相交。此外，10％没有患病的人获得了阳性检测结果。因此，10％的非病人也表现为阳性测试结果的绿色圆圈。

我们可以看到，绿色圆圈（人们测试+ ve）比蓝色圆圈（患有疾病的人）面积更大，因为5人患有疾病，其中14人获得阳性测试结果（90％ *5 + ves + 10％* 95 -ves = 14人）。

因此，测试呈阳性（绿色圆圈）的人有32.14％的几率（14人中有4.5人）患有疾病，而不是90％的我们的直觉。

## 用于分类的贝叶斯规则

**根据 “Love Life” 推断，Chris 和 Sara 谁更有可能是该邮件的作者？**

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.eab4d516ltd.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.24nhdppjeni.png)

## 后验概率

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.7fni2a5yfd.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.sprnhzotig.png)

## 为何朴素贝叶斯很朴素

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.cf7pknyj74o.png)

A, B标签是隐藏的，提供证明A, B的特征，把每个看到的词语的所有特征相乘，对A, B分别计算，用相应的先验概率，得出乘积，可以得到是否相信这个人是A 或者B的概率。 这就是朴素贝叶斯，之所以叫朴素贝叶斯，是因为它忽略了一个因素，这个因素是？

- 词序

# Mini-project 

我们有一组邮件，分别由同一家公司的两个人Sara 和Chris 各自撰写其中半数的邮件。我们的目标是仅根据邮件正文区分每个人写的邮件。

我们会先给你一个字符串列表。每个字符串代表一封经过预处理的邮件的正文；然后，我们会提供代码，用来将数据集分解为训练集和测试集。

然后使用机器学习算法，根据作者对电子邮件作者 ID进行分类。

- Sara has label 0
- Chris has label 1

*naive_bayes/nb_author_id.py*

## Sklearn 中使用 [Naive Bayes](<https://scikit-learn.org/stable/modules/naive_bayes.html#naive-bayes>)

```
from sklearn.naive_bayes import GaussianNB
from time import time
from collections import Counter


t0 = time()
clf = GaussianNB()
clf_model = clf.fit(features_train, labels_train)
print("training time:", round(time()-t0, 3), "s")

t1 = time()
pred = clf.predict(features_test)
print("predicting time:", round(time()-t1, 3), "s")

print("准确率", clf.score(features_test, labels_test))
print(clf_model)

# 统计预测值次数
print(Counter(pred).items())
```

```
training time: 8.86 s
predicting time: 0.523 s
准确率 0.9732650739476678
GaussianNB(priors=None, var_smoothing=1e-09)
dict_items([(0, 852), (1, 906)])
```
