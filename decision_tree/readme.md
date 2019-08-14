# Decision Trees 决策树

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.a95fldb11ar.png)

这个数据集是线性可分的吗？

- □ 是
- □ 不是

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.k2zokdo8gqe.png)

## 在这两种结果中，有一种结果已经代表了最终分类的结果，你觉得选择哪个就能得到正确答案？
![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.5hei2tg046.png)


![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.49m82peluzz.png)



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.8e4hq6qu05p.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ie91d1yv3s.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.5cc4jhtpc2e.png)

## 最小样本分割

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.gf7j0duavmc.png)

## 决策树参数

**将参数`min_samples_split`设置为2（默认）时，这些结点哪一个无法继续分割？**

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.iyu6hpim5h.png)

## 决策树准确性

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.gythkelu17k.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.9bxm77a0avd.png)

**这两组例子中，就只有一种分类标签而言，哪组具有更高的纯度？**

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.ic7gp2xmmu7.png)

## 信息熵（entropy）

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.q3vvgxo4fdg.png)

某些来源使用其他的对数底（例如，它们可能使用对数底 10 或底为大约 2.72 的自然对数）——这些细节可能会改变你可以获得的熵的最大值。在我们的情况中（有 2 个类），我们使用的对数底为 2 的公式将具有最大值 1。

实际上，在使用决策树时，很少需要处理对数底的细节——这里的结论是，较低的熵指向更有条理的数据，而且决策树将此用作事件分类方式。



**它表示了信息的不确定度**。是一系列样本中的不纯度的测量值。
计算信息熵的数学公式：

$$
\text {Entropy}(t)=-\sum_{i=0}^{c-1} p(i | t) \log _{2} p(i | t)
$$

$$p(i | t)$$ 代表了节点 $t$ 为分类 $i$ 的概率，其中 $\log _{2}$ 为取以 2 为底的对数。



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/entropy.ogagqfu9qmc.png)

## 信息增益 （Information Gain）

![1565689354215](1565689354215.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/iformation_gain.3g0gg972egn.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/iformation_gain1.80nsz6b7yvw.png)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/iformation_gain2.vzu60453jk.png)



# Mini-project

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

