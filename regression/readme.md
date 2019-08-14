# Regression 回归

## 连续还是离散

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.d9tfejoyo65.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.o2z3s2v4wys.png)

## 连续特征

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.6d9zsoh8fhf.png)

## 具有连续输出的监督学习

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ar7o4vh15a5.png)

## Slop and Intercept 斜率和截距

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.l2qsz1ssl5.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.yzh75pilopi.png)

## Sklearn 中使用线性回归

*regression/ages_net_worths.py*


```
import numpy
import random


def ageNetWorthData():
    random.seed(42)
    numpy.random.seed(42)

    ages = []
    for ii in range(100):
        ages.append(random.randint(20, 65))
    net_worths = [ii * 6.25 + numpy.random.normal(scale=40.) for ii in ages]
    # need massage list into a 2d numpy array to get it to work in LinearRegression
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    from sklearn.model_selection import train_test_split
    ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths)

    return ages_train, ages_test, net_worths_train, net_worths_test

```
*regression/studentRegression.py*
``` 
def studentReg(ages_train, net_worths_train):
    # import the sklearn regression module, create, and train your regression
    # name your regression reg

    # your code goes here!

    from sklearn.linear_model import LinearRegression
    reg = LinearRegression()
    reg.fit(ages_train, net_worths_train)

    return reg
```
*regression/studentMain.py*
```
#!/usr/bin/python

import numpy
import matplotlib


import matplotlib.pyplot as plt
from studentRegression import studentReg

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

reg = studentReg(ages_train, net_worths_train)

# plt.clf()
plt.scatter(ages_train, net_worths_train, color="b", label="train data")
plt.scatter(ages_test, net_worths_test, color="r", label="test data")
plt.plot(ages_test, reg.predict(ages_test), color="black")
plt.legend(loc=2)
plt.xlabel("ages")
plt.ylabel("net worths")
plt.savefig("linear_regression.png")
plt.show()
```
![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.1kmb7mj75w5.png)

**regression/regressionQuiz.py*


```
import pprint

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

# get Katie's net worth (she's 27)
# sklearn predictions are returned in an array, so you'll want to index into
# the output to get what you want, e.g. net_worth = predict([[27]])[0][0] (not
# exact syntax, the point is the [0] at the end). In addition, make sure the
# argument to your prediction function is in the expected format - if you get
# a warning about needing a 2d array for your data, a list of lists will be
# interpreted by sklearn as such (e.g. [[27]]).
# km_net_worth = 1.0 ### fill in the line of code to get the right value
km_net_worth = reg.predict([[27]])[0][0]

# get the slope
# again, you'll get a 2-D array, so stick the [0][0] at the end
# slope = 0. ### fill in the line of code to get the right value
slope = reg.coef_[0][0]

# get the intercept
# here you get a 1-D array, so stick [0] on the end to access
# the info we want
# intercept = 0. ### fill in the line of code to get the right value
intercept = reg.intercept_[0]

# get the score on test data
test_score = 0.  ### fill in the line of code to get the right value
test_score = reg.score(ages_test, net_worths_test)

# get the score on the training data
training_score = 0.  # fill in the line of code to get the right value
training_score = reg.score(ages_train, net_worths_train)


def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth": km_net_worth,
            "slope": slope,
            "intercept": intercept,
            "stats on test": test_score,
            "stats on training": training_score}


if __name__ == "__main__":
    pprint.pprint(submitFit())
```

```
{'slope': 6.473549549577059,
'stats on training': 0.8745882358217186, 
'intercept': -14.35378330775552, 
'stats on test': 0.812365729230847, 
'networth': 160.43205453082507}
```

## 线性回归误差

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/error.zxdoxvhg9xi.png)

## 误差和拟合质量

## 拟合能将哪种误差降至最低？

- □ 第一个和最后一个数据点的误差

- □ 所有数据点的误差和

- □ 所有误差的绝对值的和

- □ 所有误差的平方和

  > - 所有误差的绝对值的和
  > - 所有误差的平方和

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.h2ha51a4j7h.png)

## 最小化误差平方和

- 最小二乘法
- 梯度下降法

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.3a8a1rp9ygc.png)

## 最小化绝对误差的问题

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.bnf96tit12h.png)

- 使用误差平方和查找回归的方法，能使回归（算法）的实现简单很多，最小化误差平方和，而不是绝对误差时，更容易找到回归线。

## SSE的问题

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ixnzthedqfs.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.v8v1g1w148.png)



平方偏差和的一个不足之处，添加的数据越多，平方误差和几乎必定会增加，但并不代表拟合地不好，然而，如果对不同数量点的两个数据进行比较，如果使用平方误差和来拟合更好地图像，就会出现很大的问题。平方误差和会因为使用数据点的数量出现偏差，尽管拟合得问题不大。

## 回归的R平方指标

**R平方的优点是与训练点数量无关，始终在0到1之间**，数字越小（越接近0）意味着回归线在捕捉数据趋势方面表现不佳，数字越大（越接近1），回归线在描述输入（x变量）与输出（y变量）之间的关系方面表现较好。所以，它比误差平方和更可靠一些，尤其在数据集中的数据数可能会改变时。

```
test_score = reg.score(ages_test, net_worths_test)
```



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.hxha1rwrm1.png)

## 什么数据适用于线性回归

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.0yoapxbhxlfh.png)





## 比较分类与回归

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.nr4jzk0phr9.png)



- 分类：离散的，寻找决策边界。使用查准率作为指标。
- 回归：持续的，通常使用回归预测数字。寻找最优拟合线，是拟合数据的线条，而不是数据边界。使用R平方或误差平方和作为指标。

## 多元回归

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.7xclkccubqk.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.1dv5szvk5f6.png)

# Mini-project 

*regression/finance_regression.py*

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.63zb801641e.png)

从 Python 3.3 版本开始，字典的键值顺序有所改变，在每次代码运行时，字典的键值皆为随机排序。这会让我们在 Python 2.7 环境下工作的评分者遭遇一些兼容性的问题。为了避免这个问题，请在 `finance_regression.py` 文件的第26行 `featureFormat` 调用时添加一个参数

```
sort_keys = '../tools/python2_lesson06_keys.pkl'
```

从 sklearn 导入 *LinearRegression* 并创建/拟合回归。将其命名为 reg，这样绘图代码就能将回归覆盖在散点图上呈现出来。回归是否大致落在了你期望的地方？

提取斜率（存储在 *reg.coef_* 属性中）和截距。

- 斜率和截距是多少？

假设你是一名悟性不太高的机器学习者，你没有在测试集上进行测试，而是在你用来训练的相同数据上进行了测试，并且用到的方法是将回归预测值与训练数据中的目标值（比如：奖金）做对比。

- 你找到的分数是多少？

现在，在测试数据上计算回归的分数。

- 测试数据的分数是多少？如果只是错误地在训练数据上进行评估，你是否会高估或低估回归的性能？

```
features_list = ["bonus", "salary"]

from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(feature_train, target_train)
reg.predict(feature_test)
print(f'斜率:{reg.coef_}, 截距:{reg.intercept_}, '
      f'训练集分数: {reg.score(feature_train, target_train)}, '
      f'测试集分数: {reg.score(feature_test, target_test)}')
```

```
斜率:[5.44814029], 
截距:-102360.54329387983, 
训练集分数: 0.04550919269952436, 
测试集分数: -1.48499241736851
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.5nveenixwko.png)

我们有许多可用的财务特征，就预测个人奖金而言，其中一些特征可能比余下的特征更为强大。例如，假设你对数据做出了思考，并且推测出“long_term_incentive”特征（为公司长期的健康发展做出贡献的雇员应该得到这份奖励）可能与奖金而非工资的关系更密切。

证明你的假设是正确的一种方式是根据长期激励回归奖金，然后看看回归是否显著高于根据工资回归奖金。

- 根据**长期奖励回归奖金**—测试数据的分数是多少？

```
features_list = ["bonus", "long_term_incentive"]
```

```
斜率:[1.19214699], 
截距:554478.7562150093, 
训练集分数: 0.21708597125777662, 
测试集分数: -0.5927128999498639
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.xfpfhv3mdt.png)

- 如果你需要预测某人的奖金，你是通过他们的工资还是长期奖金来进行预测呢？

>long_term_incentive



## 异常值破坏回归

这是下节课的内容简介，关于异常值的识别和删除。返回至之前的一个设置，你在其中使用**工资预测奖金**，并且重新运行代码来回顾数据。你可能注意到，少量数据点落在了主趋势之外，即某人拿到高工资（超过 1 百万美元！）却拿到相对较少的奖金。此为异常值的一个示例，我们将在下节课中重点讲述它们。

类似的这种点可以对回归造成很大的影响：如果它落在训练集内，它可能显著影响斜率/截距。如果它落在测试集内，它可能比落在测试集外要使分数低得多。就目前情况来看，此点落在测试集内（而且最终很可能降低分数）。让我们做一些处理，看看它落在训练集内会发生什么。在 *finance_regression.py* 底部附近并且在 `plt.xlabel(features_list[1])` 之前添加这两行代码：
```
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
```
现在，我们将绘制两条回归线，一条在测试数据上拟合（有异常值），一条在训练数据上拟合（无异常值）。来看看现在的图形，有很大差别，对吧？单一的异常值会引起很大的差异。

- 新的回归线斜率是多少？

```
features_list = ["bonus", "salary"]
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
```

```
无异常值斜率:[2.27410114],
截距:124444.38886605436,
训练集分数: -0.12359798540343814,
测试集分数: 0.251488150398397
```

```
plt.plot(feature_test, reg.predict(feature_test), color="r")
plt.plot(feature_train, reg.predict(feature_train), color="b")
```

