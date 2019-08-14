# 异常值 Outliers

## 产生异常值的原因

- 传感器故障
- 数据输入错误。手动输入数据，很有可能出现数字错误
- 外部数据
- 反常事件

## 选择异常值

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.y33enfut0wj.png)

## 异常值检测/删除算法

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.fhd8iuogb2c.png)

# Mini-project

## 带有异常值的回归斜率

Sebastian 向我们描述了改善回归的一个算法，你将在此项目中实现该算法。你将在接下来的几个测试题中运用这一算法。总的来说，你将在所有训练点上拟合回归。舍弃在实际 y 值和回归预测 y 值之间有最大误差的 10% 的点。

先开始运行初始代码 (*outliers/outlier_removal_regression.py*) 和可视化点。一些异常值应该会跳出来。部署一个线性回归，其中的净值是目标，而用来进行预测的特征是人的年龄（记得在训练数据上进行训练！）。

数据点主体的正确斜率是 6.25（我们之所以知道，是因为我们使用该值来生成数据）；

- 你的回归的斜率是多少？

- 当使用回归在测试数据上进行预测时，你获得的分数是多少？

```
reg = LinearRegression()
reg.fit(ages_train, net_worths_train)
reg.predict(ages_test)
print(f'斜率:{reg.coef_}, 截距:{reg.intercept_}, '
      f'测试集分数: {reg.score(ages_test, net_worths_test)}')
```

```
斜率:[[5.07793064]], 截距:[25.21002155], 测试集分数: 0.8782624703664671
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/before_clean.93qo0ibespi.png)

## 清理后的斜率

你将在 *outliers/outlier_cleaner.py* 中找到 *outlierCleaner()* 函数的骨架并向其填充清理算法。用到的三个参数是：*predictions* 是一个列表，包含回归的预测目标；*ages* 也是一个列表，包含训练集内的年龄；*net_worths* 是训练集内净值的实际值。每个列表中应有 90 个元素（因为训练集内有 90 个点）。你的工作是返回一个名叫cleaned_data 的列表，该列表中只有 81 个元素，也即预测值和实际值 (net_worths) 具有最小误差的 81 个训练点 (90 * 0.9 = 81)。cleaned_data 的格式应为一个元组列表，其中每个元组的形式均为 (age, net_worth, error)。

- 一旦此清理函数运行起来，你应该能看到回归结果发生了变化。新斜率是多少？（是否更为接近 6.25 这个“正确”结果？）

- 当使用回归在测试集上进行预测时，新的分数是多少？

```
def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """

    cleaned_data = []
    # your code goes here
    for i in range(len(predictions)):
        cleaned_data.append((ages[i], net_worths[i],
                             abs(predictions[i] - net_worths[i])))
    # 根据错误值排序
    cleaned_data.sort(key=lambda x: x[2])

    # 删除具有最大残差的 10% 的点
    return cleaned_data[:int(len(ages) * 0.9)]

cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train)

if len(cleaned_data) > 0:
    ages, net_worths, errors = list(zip(*cleaned_data))
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))  
```

```
清理后
斜率:[[6.36859481]], 截距:[-6.91861069], 测试集分数: 0.9831894553955322
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/after_clean.g3b17ga0syd.png)

## 安然异常值

在本节回归课程的迷你项目中，你使用回归来预测安然雇员的奖金。如你所见，单一的异常值都可以对回归结果造成很大的差异。但是，我们之前没有跟你说过的是，你在项目中使用的数据集已经被清理过明显的异常值了。第一次看到数据集时，识别并清除异常值是你一直应该思考的问题，而你现在已经通过安然数据有了一定的实践经验。

你可以在 *outliers/enron_outliers.py* 中找到初始代码，该代码读入数据（以字典形式）并将之转换为适合 sklearn 的 numpy 数组。由于从字典中提取出了两个特征（“工资”和“奖金”），得出的 numpy 数组维度将是 N x 2，其中 N 是数据点数，2 是特征数。对散点图而言，这是非常完美的输入；我们将使用 matplotlib.pyplot 模块来绘制图形。（在本课程中，我们对所有可视化均使用 pyplot。）将这些行添加至脚本底部，用以绘制散点图： 

```
import matplotlib.pyplot as plt

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.savefig("bonus-salary-outliers.png")
plt.show()
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/bonus-salary-outliers.5y0t92nm56a.png)

## 识别最大的安然异常值

有一个异常值应该会立即跳出来。现在的问题是识别来源。我们发现原始数据源对于识别工作非常有帮助；你可以在 *final_project/enron61702insiderpay.pdf* 中找到该 PDF。

- 该数据点的字典键名称是什么？（例如：如果是 Ken Lay，那么答案就是“LAY KENNETH L”）。

```
def remove_nan(feature):
    print("寻找最大奖金")
    feature_no_nan = []
    for i, item in data_dict.items():
        if item[feature] != "NaN":
            feature_no_nan.append((i, item[feature]))
    feature_no_nan.sort(key=lambda x: x[1], reverse=True)
    return feature_no_nan
    
print(remove_nan("bonus"))
```

```
[('TOTAL', 97343619), ('LAVORATO JOHN J', 8000000), ('LAY KENNETH L', 7000000), ('SKILLING JEFFREY K', 5600000), ('BELDEN TIMOTHY N', 5249999), ('ALLEN PHILLIP K', 4175000), ...]
```

## 移除安然异常值？
**你认为这个异常值应该并清除，还是留下来作为一个数据点？**

- 留下来，它是有效的数据点
- 清除掉，它是一个电子表格 bug（select）
- 清除掉，它是一个拼写错误

## 还有更多异常值吗?

从字典中快速删除键值对的一种方法如以下行所示：

*dictionary.pop( key )*

写下这样的一行代码（你必须修改字典和键名）并在调用 *featureFormat()* 之前删除异常值。然后重新运行代码，你的散点图就不会再有这个异常值了。

```
data_dict.pop("TOTAL")
data = featureFormat(data_dict, features)
```

所有异常值都没了吗？

- Enron 数据中还有异常值吗？ 

可能还有四个

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/bonus-salary-outliers.9uo2wzsfz3b.png)

## 再识别两个异常值

我们认为还有 4 个异常值需要调查；让我们举例来看。两人获得了至少 5 百万美元的奖金，以及超过 1 百万美元的工资；换句话说，他们就像是强盗。

和这些点相关的名字是什么？与当前 Enron 异常值相关联的名称有哪些？（给出字典 key 值中所写的名称 – 如：Phillip Allen 将是 ALLEN PHILLIP K）

```
('LAVORATO JOHN J', 8000000), ('LAY KENNETH L', 7000000)
```

## 移除这些异常值？

你是否会猜到这些就是我们应该删除的错误或者奇怪的电子表格行，你是否知道这些点之所以不同的重要原因？（换句话说，在我们试图构建 POI 识别符之前，是否应该删除它们？）

- 留下来，它是有效的数据点。他们是安然公司最大的两个老板，绝对是有兴趣的人。

