# 安然数据集

POI (Person Of Interest) 指的是在这个问题中的相关人物。

建议你在有时间的时候，花一个半小时，看一下**【纪录片】安然：房间里最聪明的人 Enron: The Smartest Guys in the Room** ([Bilibili 链接](http://www.bilibili.com/video/av10093141/) 或 [Youtube链接](https://www.youtube.com/watch?v=rDyMz1V-GSg) )来了解这个事件。

Katie 从 [这篇新闻](http://usatoday30.usatoday.com/money/industries/energy/2005-12-28-enron-participants_x.htm) 以及很多其他新闻中，搜集并列出了所有她认为可能与安然丑闻事件有关的人的名字。你可能有些困惑名字旁边的 Y 和 N 代表什么，别担心，在后面会提到。

**下载数据**

目前从[CMU 安然数据集页面](https://www.cs.cmu.edu/~./enron/) 能下载到的最新数据集为 **[May 7, 2015 Version of dataset](https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz)**.

安然欺诈案是一个混乱而又引人入胜的大事件，从中可以发现几乎所有想像得到的企业违法行为。安然的电子邮件和财务数据集还是巨大、混乱的信息宝藏，而且，在你稍微熟悉这些宝藏后，它们会变得更加有用。我们已将这些电子邮件和财务数据合并为一个数据集，而你将在此迷你项目中研究它。

## Mini-project1 : 研究安然数据集

*datasets_questions/explore_enron_data.py*

```
安然数据集大小: 146
安然数据集中特征数量: 21

安然数据集中特征
['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'email_address', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi', 'poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock', 'director_fees']

安然数据集中嫌疑人数量: 18
嫌疑人(POI)数量及名单
['BELDEN TIMOTHY N', 'BOWEN JR RAYMOND M', 'CALGER CHRISTOPHER F', 'CAUSEY RICHARD A', 'COLWELL WESLEY', 'DELAINEY DAVID W', 'FASTOW ANDREW S', 'GLISAN JR BEN F', 'HANNON KEVIN P', 'HIRKO JOSEPH', 'KOENIG MARK E', 'KOPPER MICHAEL J', 'LAY KENNETH L', 'RICE KENNETH D', 'RIEKER PAULA H', 'SHELBY REX', 'SKILLING JEFFREY K', 'YEAGER F SCOTT']


poi_names.txt中嫌疑人数量为:35人
poi_names.txt中嫌疑人中安然的员工有4人


James Prentice 名下的股票总值是多少？ 1095040
我们有多少来自 Wesley Colwell 的发给嫌疑人的电子邮件？ 11
Jeffrey Skilling 行使的股票期权价值是多少？ 19250000
```



### 研究安然欺诈案

以下哪个计谋 Enron 未参与？

- 在每月月末将资产出售给空壳公司，然后在下月初买回，以隐藏会计损失
- 导致加利福尼亚的电网故障
- 非法获取了一份政府报告，使他们得以垄断冷冻浓缩橙汁期货
- 主谋为一位沙特公主快速获得了美国国籍
- 与百视达 (Blockbuster) 影业合作，在互联网上通过流媒体传播影片

-  欺诈案发生的多数时间内，安然的 CEO 是谁？

effrey K. Skilling

- 安然的董事会主席是谁？
Kenneth L. Lay

- 欺诈案发生的多数时间内，安然的 CFO（首席财务官）是谁？
Andrew S. Fastow

- 这三个人（Lay、Skilling 和 Fastow）当中，谁拿回家的钱最多（`total_payments`特征的最大值）？

>安然的 CEO Jeffrey K. Skilling 拿了多少钱？8682716
安然的 董事会主席 Kenneth L. Lay 拿了多少钱？103559793
安然的 CFO（首席财务官） Andrew S. Fastow 拿了多少钱？2424083

> 当特征没有明确的值时，我们使用什么来表示它？NaN

> 此数据集中有多少雇员有量化的工资？51
> 此数据集中已知的邮箱地址是否可用？35

> 整个数据集中 total_payments被设置为`NaN`的数量为21，比例为0.14
> POI中 total_payments被设置为`NaN`的数量为0，比例为0.0

### 假设添加部分值

如果你再次添加了全是 POI 的 10 个数据点，并且对这些雇员的薪酬总额设置了`NaN`，你刚才计算的数字会发生变化。

> 人的数量变成了多少？156
> 薪酬总额被设置了“NaN”的雇员数变成了多少？31

> 数据集中 POI 的人数变成了多少？28
> POI 中，股票总值为“NaN”的人数变成了多少？10

### 混合数据源
此例中加入了新的 POI，而我们没有任何人的财务信息，这就带来了一个微妙的问题，即算法可能会注意到我们缺少他们的财务信息，并将这一点作为他们是 POI 的线索。换个角度来看，为我们的两个类生成数据的方式现在有所不同 - 非 POI 的人全都来自财务电子表格，之后手动加入了许多 POI。这种不同可能会诱使我们以为我们的表现优于实际状况 - 假设你使用 POI 检测器来确定某个未见过的新人是否是 POI，而且该人不在电子表格上。然后，他们的所有财务数据都将包含`NaN`，但该人极有可能不是 POI（世界上非 POI 的人比 POI 多得多，即使在安然也是如此）- 然而你可能会无意中将他们标识为 POI！

这就是说，在生成或增大数据集时，如果数据来自不同类的不同来源，你应格外小心。它很容易会造成我们在此展示的偏差或错误类型。可通过多种方法处理此问题。举例而言，如果仅使用了电子邮件数据，则你无需担心此问题（在这种情况下，财务数据中的差异并不重要，因为并未使用财务特征）。还可以通过更复杂的方法来估计这些偏差可能会对你的最终答案造成多大影响，不过此话题超出了本课程的范围。

目前的结论就是，要非常小心地对待引入来自不同来源（具体取决于类）的特征这个问题！引入此类特征常常会意外地带来偏差和错误。

# 回归

## Mini-project2 : 奖金目标和特征

*regression/finance_regression.py*

奖金和工资 `bonus `&  `salary`

```
reg.fit(feature_train, target_train)
plt.plot(feature_test, reg.predict(feature_test), color="r")
print(f'斜率:{reg.coef_}, 截距:{reg.intercept_}, '
      f'训练集分数: {reg.score(feature_train, target_train)}, '
      f'测试集分数: {reg.score(feature_test, target_test)}')
```

> 斜率:[5.44814029], 
>
> 截距:-102360.54329387983, 
>
> 训练集分数: 0.04550919269952436, 
>
> 测试集分数: -1.48499241736851

训练测试集，绘制训练集回归线（蓝色回归线）

```
reg.fit(feature_test, target_test)
plt.plot(feature_train, reg.predict(feature_train), color="b")
```



> 斜率:[2.27410114], 
> 截距:124444.38886605436, 
> 训练集分数: -0.12359798540343814, 
> 测试集分数: 0.251488150398397

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/bonus-salary-regression.9sbc1v9qutk.png)

奖金和长期激励回归奖金 `bonus` & `long_term_incentive he `

> 斜率:[1.19214699], 
>
> 截距:554478.7562150093, 
>
> 训练集分数: 0.21708597125777662, 
>
> 测试集分数: -0.5927128999498639

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/bonus-long_term_incentive-regression.pior439mkrk.png)

如果你需要预测某人的奖金，你是通过他们的工资还是长期奖金来进行预测呢？

- 长期奖金

# 异常值

*outliers/outlier_removal_regression.py*

### 内净值和年龄线性回归  `net_worths` & `ages`

清理异常值之前：

> 斜率:[[5.07793064]], 
>
> 截距:[25.21002155], 
>
> 测试集分数: 0.8782624703664671

清理异常值：

在 *outliers/outlier_cleaner.py* 中找到 *outlierCleaner()* 函数的骨架并向其填充清理算法。

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
```



> 斜率:[[6.36859481]], 
>
> 截距:[-6.91861069], 
>
> 测试集分数: 0.9831894553955322



### 奖金和工资 `bonus `&  `salary` 线性回归

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/bonus-salary-outliers.q226tk8u3eo.png)

查看源数据，该最大值是

('TOTAL', 97343619),

 ('LAVORATO JOHN J', 8000000), 

('LAY KENNETH L', 7000000), 

('SKILLING JEFFREY K', 5600000), 

('BELDEN TIMOTHY N', 5249999), 

('ALLEN PHILLIP K', 4175000),

删除该异常值`data_dict.pop("TOTAL")`

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/bonus-salary-outliers.tukoo46x6n.png)