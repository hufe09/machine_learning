# 评估指标

# Confusion Matrices 混淆矩阵

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.vr6iqta218m.png)

现在要讨论**混淆矩阵**，你可能未曾听说过这个名词 但如果要从事机器学习并向别人讨论这个的话，这是个很好的词汇，想象我们有一个分类器 在这条线上方 是红色的X 线下方是绿色的圈，想象我们有一个分类器 在这条线上方 是红色的X 线下方是绿色的圈，与以前一样 我们将红色 X 称为正例示例，绿环为负例，**混淆矩阵是一个二乘二的矩阵**，对照实际类 它可为正例或负例，这个概念 我们称为一个**预测类**，它是我们的输出值 也可以为正例或负例，右边的点的数量将会增加左边的计数，我们观察一下，我们在这里取一个数据点为负例，其分类位于该分类器的绿色一侧，如果要对这种类型的数据进行计数，那它会落入左侧四个位置的哪一个里？

实际类是负例的 但是分类器也将它放入负例的一侧

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.swexno6jh1.png)



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.zz9q9cgni47.png)



显然 对于误归类的点来说事情变得更加困难，所以我们来看下这里的这个点，我们来把它加入(到图里)，那它会落入左侧的什么位置？

> 这个数据点是真阳性，它是个红叉但被预测为负例，所以实际的真正例被预测为负例

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.w901m5dnd9d.png)



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.wtv31oxogz.png)



## 虚假报警

因此 在这里我有最后一个问题，假设这是一个报警系统 正向或者正例意味着，你们的房子里有一个盗贼，你们认为哪个框 什么类型的事件，最能说明报警系统的假警报的特点？

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.4fgcbq7b683.png)

我说是误报警 是因为报警响了 即正例表示了盗窃行为，但实际并没有盗贼，在真实世界中事件是负例 但系统说是正例，所以在这里要画勾 你可以看出有点不对称

可以看出你可能关心给你正例报警的负例事件，不同于被错过的正例事件，先提前给大家透露一点知识，通常 通过改变参数 可将这个曲线向两个方向变动，如果你认为那是误报警 其会导致警察来你家捉贼，其代价应不同于被错过的事件，后者的一个例子是有人抢走了你的物品 而警察却一无所知，这时 你可能希望将这条线重新上下调整

## 决策树混淆矩阵

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565621559895.tt2vbbf1es.png)

## 特征脸的混淆矩阵



比如我们查看七个不同的白人男性政治家，从 George Bush 到 Gerhard Schroeder 然后进行 EigenFace 分析，提取出此数据集的主成分 然后再使用Eigenfaces 对应着姓名重新绘制新的面部图 从而识别这些人。我们不管这些面部图，我要做的是给你一个有典型的输出，我们将使用混淆矩阵学习输出

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ozkcoujanp.png)

我们在左侧放置正确的姓名 正确的类型标签，在上方放置预测的内容，比如说 这里的数字 1 对应的是 Donald Rumsfeld，但被错误地认为是 Colin Powell![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.8mb3359nugg.png)

首先 一个简单的问题，这七个政治家中 哪个在我们的数据集中出现次数最多？　

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.0js178l6l9e.png)



第二个问题Gerhard Schroeder在我们的数据集中有多少张图像？![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.azcku14x55w.png)

预测这一边有多少认为是Gerhard Schroeder图像的预测？![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.5aqos2swrbm.png)

答案是 15，就是这一列 14 加 1，这让你多少能了解这些数据，很明显 在混淆矩阵中，如果主对角线上有最大值 你会十分高兴，因为所有非对角线元素被错误分类![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.i30ja3b692.png)



正确分类Hugo，学习算法正确分类Hugo的概率？![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.agdcdhnq7we.png)

现在我问个不同的问题，假如你的预测器说这是 Hugo Chavez，运行学习算法 算法也将它分类为 Hugo Chavez，那么这个人真的是 Hugo Chavez 的概率有多大？

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.6in5i1lt3a.png)

概率恰好是 1，似乎结果完美地正确预测了 Hugo Chavez，如果你查看所有由算法识别出的 Hugo Chavez 的图像案例，10 个里有 10 个都是对的结果，所以 10 除以 10 等于 1



## 查全率和查准率 Precision and Recall

我们来看两个专业术语，**查全率和查准率**，例如 假设此人就是 Hugo Chavez

那么 Hugo Chavez 的查全率就是准确识别的次数比率,即在这个人是 Hugo Chavez 的情况下 我们的算法准确识别 Hugo Chavez 的概率,因此 我们得出的结果是 10/16

查准率也是比率概念,是我们的算法预测的概率 是我们学习的第二个比率,这里恰好是 1,假设我们的算法检测到了 Hugo Chavez,那么此人确实是 Hugo Chavez 的概率,这里的查准率刚好是 10/10

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.td0t0i3s8j.png)



True Positives in Eigenfaces

False Positives in Eigenfaces

False Negatives in Eigenfaces

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.17g0wak374l.png)



### Equation for Precision 查全率方程

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.kiimi6edl1.png)



### Equation for Recall 查准率方程

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.v8rf2emrd2l.png)

## 分类器的性能

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.vr6iqta218m.png)

- TN: True Negative  预测正确且预测类型为Negative 
- FP:  False Positive  预测错误且预测类型为Positive  
- FN:  False Negative  预测错误且预测类型为Negative
- TP:  True Positive  预测正确且预测类型为Positive

**Model Performance**

- 准确率：

$$
\text { Accuracy }=\frac{T P+T N}{T P+T N+F P+F N}
$$

- 精确度：

$$
\text { Precision }=\frac{T P}{T P+F P}
$$

精度是指在所有预测为正例的分类中，预测正确的程度为正例的效果。
**精度越高越好**。

- 召回率：

$$
\text { Recall }=\frac{T P}{T P+F N}
$$

召回率是指在所有预测为正例（被正确预测为真的和没被正确预测但为真的）的分类样本中，召回率是指预测正确的程度。它，也被称为敏感度或真正率（TPR）。
**召回率越高越好**。

- F1值：

$$
F_{1}=\left(\frac{2}{\operatorname{recall}^{-1}+\text { precision }^{-1}}\right)=2 \cdot \frac{\text { precision } \cdot \text { recall }}{\text { precision }+\text { recall }}
$$

通常实用的做法是将精度和召回率合成一个指标F-1值更好用，特别是当你需要一种简单的方法来衡量两个分类器性能时。F-1值是精度和召回率的调和平均值。

- F2值：

$$
\text { F2 }=\frac{(1+2+2) * Precision * Recall}{4 * Precision + Recall}
$$

**假正例（I型错误）**——原假设正确而拒绝原假设

**假负例（II型错误）**——原假设错误而接受原假设



**Recall**: True Positive / (True Positive + False Negative). 在所有TP的项目中，有多少被正确归类为阳性的。或者简单地说，从数据集中“召回”了多少阳性的项目。

**Precision**: True Positive / (True Positive + False Positive). 在所有标为阳性的项目中，有多少真正属于阳性类别。


![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565620081932.0b76lsr2ddar.png)

# MIni-project

*evaluation/evaluate_poi_identifier.py*

## 将度量标准应用于POI标识符

在 *validation/validate_poi.py* 中，POI标识符的准确度（在测试集上）为 0.724。

**测试集中的POI数量？**

```
count = 0
for i in labels_test:
    if i == 1:
        count += 1

print("Number of POIs in test set is", count)
```

```
Number of POIs in test set is 4
```

**测试集中有多少人？**

```
print("Number of people in test set is", len(labels_test))
```

```
Number of people in test set is 29
```

**如果将测试集中的每个人预测定义为0.（不是POI），那么它的准确度是多少？**

0

**Number of True Positives?**

```
def evaluation_metrics(truth, pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for i in range(len(truth)):
        if truth[i] == 1 and pred[i] == 1:
            TP += 1
        if truth[i] == 1 and pred[i] == 0:
            FN += 1
        if truth[i] == 0 and pred[i] == 1:
            FP += 1
        if truth[i] == 0 and pred[i] == 0:
            TN += 1
    model_metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "F1": None,
        "F2": None
    }
    try:
        total = TP + TN + FP + FN
        print("Total", total, "TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)

        accuracy = 1.0 * (TP + TN) / total
        model_metrics["accuracy"] = accuracy
        precision = 1.0 * TP / (TP + FP)
        model_metrics["precision"] = precision
        recall = 1.0 * TP / (TP + FN)
        model_metrics["recall"] = recall
        f1 = 2.0 * TP / (2 * TP + FP + FN)
        f1 = 2.0 * (precision * recall)/(precision + recall)
        model_metrics["F1"] = f1
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        model_metrics["F2"] = f2
    except:
        print("Got a divide by zero when trying out the set.")
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")

    return TP, TN, FP, FN, model_metrics
    
TP, TN, FP, FN, model_metrics = evaluation_metrics(labels_test, pred)
pprint.pprint(model_metrics)
```

```
Total 29 TP: 0 TN: 21 FP: 4 FN: 4

Got a divide by zero when trying out the set.
Precision or recall may be undefined due to a lack of true positive predicitons.

{'F1': None,
 'F2': None,
 'accuracy': 0.7241379310344828,
 'precision': 0.0,
 'recall': 0.0}
```

**使用*precision_score*和*recall_score*提供*sklearn.metrics*计算的数量**

```
def sklearn_metrics(truth, prediction):
    from sklearn import metrics

    print("Precision score", metrics.precision_score(truth, prediction))
    print("Accuracy score", metrics.accuracy_score(truth, prediction))
    print("Recall score", metrics.recall_score(truth, prediction))
    print("F1 score", metrics.f1_score(truth, prediction))
```

```
Precision score 0.0
Accuracy score 0.7241379310344828
Recall score 0.0
F1 score 0.0
```

# How Many True Positives?

```
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]

true labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
```

**How many true positives are there?**

> 6

**How many true negatives are there in this example?**

> 2

**How many false positives are there?**

> 3

**How many false negatives are there?**

> 9

**What's the precision of this classifier?**

> 6/(6+2) = 0.75

**What's the recall of this classifier?**

> 6/(6+3) = 0.667



```
predictions = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1]
true_labels = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]

TP, TN, FP, FN, model_metrics = evaluation_metrics(true_labels, predictions)

pprint.pprint(model_metrics)
sklearn_metrics(true_labels, predictions)
```

```
Total 20 TP: 6 TN: 9 FP: 3 FN: 2

{'F1': 0.7058823529411765,
 'F2': 0.7317073170731707,
 'accuracy': 0.75,
 'precision': 0.6666666666666666,
 'recall': 0.75}
 
Precision score 0.6666666666666666
Accuracy score 0.75
Recall score 0.75
F1 score 0.7058823529411765
```



## 理解指标

“My true positive rate is high, which means that when a **POI** is present in the test data, I am good at flagging him or her.”

我的真阳率很高，这意味着当测试数据中存在**POI**时，我擅长标记他或她。



“My identifier doesn’t have great **precisionl**, but it does have good **recall**. That means that, nearly every time a POI shows up in my test set, I am able to identify him or her. The cost of this is that I sometimes get some false positives, where non-POIs get flagged.”

我的标识符没有很高的**精确度**，但是它有很好的**召回率**。这意味着，几乎每次POI出现在我的测试集中时，我都能识别他或她。 这样做的代价是我有时会得到一些误报，其中非POI会被标记。



“My identifier doesn’t have great **recall**, but it does have good **precision**. That means that whenever a POI gets flagged in my test set, I know with a lot of confidence that it’s very likely to be a real POI and not a false alarm. On the other hand, the price I pay for this is that I sometimes miss real POIs, since I’m effectively reluctant to pull the trigger on edge cases.”

我的标识符没有很大的**召回率**，但它有很好的**精确度**。 这意味着每当POI在我的测试集中被标记时，我就会非常自信地知道它很可能是真正的POI而不是虚警。 另一方面，我为此付出的代价是我有时会错过真正的POI，因为我实际上不愿意触发边缘情况。



“My identifier has a really great **F1 score**.

This is the best of both worlds. Both my false positive and false negative rates are **low**, which means that I can identify POI’s reliably and accurately. If my identifier finds a POI then the person is almost certainly a POI, and if the identifier does not flag someone, then they are almost certainly not a POI.”

我的标识符有一个非常棒的**F1值**。这是两全其美的。 我的假阳率和假阴率都很低，这意味着我可以可靠而准确地识别POI。 如果我的标识符找到POI，则该人几乎肯定是POI，如果标识符没有标记某人，那么他们几乎肯定不是POI。“ 



>For an explanation of F1 score, check out [this link](https://en.wikipedia.org/wiki/F1_score)