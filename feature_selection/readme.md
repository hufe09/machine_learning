# 特征选择 Feature Selection

> make everything as simple as possible, but not simple.  -- Albert Einstein

## 为何使用特征选择
你的机器学习算法的上限，就是你放入的特征

你应该有在可选选项中找到最佳特征的方法，也就是抛弃对你不是有真正帮助的东西，可能你的数据中有些模式很有用，所以你可以通过人类的直觉对新加入的特征感知模式

## 向安然数据集加入新特征
在编写新特征时，遵循以下原则：

- 利用人的直觉
你认为什么特征可能包含一些利用机器学习算法的模式
比如我对安然数据集有这样的直觉：嫌疑人间相互发邮件的频率很高
- 编写新特征
- 可视化
可视化结果会让你知道你的进展方向是否正确，你的新特征是否会帮助你鉴别我们想要解决的分类问题
- 重复
如果你想要找到对你最有用的新特征是什么，经常要多次重复这个过程

## 示例：有漏洞的特征

Katie 在处理安然 POI 识别符时，设计了一个特征，用来对既定人员以 POI 身份出现在相同的邮件中时进行识别。 举一个例子，如果 Ken Lay 和 Katie Malone 是同一封邮件的接收人，那么 Katie Malone 的“shared receipt”特征就应该得到递增。 如果她与 POI 共享很多邮件，那么她自己很可能就是一个 POI。

这里有一个小漏洞，即当这种情况发生时，Ken Lay 的 `shared receipt` 计数器也应该得到递增。 当然，Ken Lay 总是与 POI 共享收据，因为他就是 POI。 因此，“shared receipt”特征在查找 POI 方面就变得异常强大，因为它将每个人的标签有效编成了一个特征。

我们最初发现这一点，是因为我们对总是返回 100% 准确率的一个分类器感到怀疑。于是，我们每次删除一个特征，然后就发现了这一特征是所有这一切的根源。 我们在回顾特征代码时发现了以上提到的漏洞。 我们更改了代码，以便仅当不同的 POI 收到邮件时，该既定人员的 `shared receipt` 特征才会得到递增，重新运行代码，然后再次尝试。准确率落到了更为合理的水平。

我们从这个例子中认识到：

任何人都有可能犯错—要对你得到的结果持怀疑态度！

你应该时刻警惕 100% 准确率。不寻常的主张要有不寻常的证据来支持。

如果有特征过度追踪你的标签，那么它很可能就是一个漏洞！

如果你确定它不是漏洞，那么你很大程度上就不需要机器学习了——你可以只用该特征来分配标签。

## 特征删除

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.25oyckpao5wi.png)

- 特征太杂乱了，很难分辨它是否能可靠地帮助你测量你想测量的东西
- 由于某种原因，特征可能会导致你的模型过度拟合
- 这个特征与当前已存在的特征密切关联，高度相关，它只是在不停地向你提供重复的信息，是当前其他特征也有提供的信息
- 新特征可能会拖慢训练/测试过程，为了让所有的东西都快速运转，只保留最低的必要的特征数量，以便达到最好的效果

## 特征不等于信息


想从数据中获取的是信息，要得出结论，形成见解。

特征是特定的试图获取信息的数据点的实际数量或特点，和信息本身不同。

总体来说，我们需要的就是，用尽量少的特征得到尽量多的信息。

如果这个特征不能给予我们信息，就要删除它，因为它可能会引起过拟合或者其他 bug。

在 sklearn 中自动选择特征有多种辅助方法。多数方法都属于单变量特征选择的范畴，即独立对待每个特征并询问其在分类或回归中的能力。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.gh34ebjmygf.png)

## 单变量特征选择

sklearn 中有两大单变量特征选择工具：SelectPercentile 和 SelectKBest。 两者之间的区别从名字就可以看出：SelectPercentile 选择最强大的 X 特征（X 是参数），而 SelectKBest 选择 K 个最强大的特征（K 是参数）。

由于数据维度太高，特征约简显然要用到文本学习。 实际上，在最初的几个迷你项目中，我们就已经在 Sara/Chris 邮件分类的问题上进行了特征选择；你可以在 *tools/email_preprocess.py* 的代码中看到它。

## 偏差、方差和特征数量

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.n3u8cmibv5g.png)

特征数量越少，就会进入很经典的一类高偏差型领域。

非常典型的情况是，可能需要几个特征，来完整描述一个数据集中的所有模式，但仅仅用了其中的一两个，这就会使我们处于一种不会真正对数据足够关心的状态，这是一个简化的情况，换句话说就是**高偏差**。

如果使用很多的特征获得拟合程度最好的回归，或者分类。这就是经典的**高方差**情形。造成过拟合。

最终我们需要的是：
使用较少的几个特征来拟合算法，但同时，就回归而言，希望得到较大的R²，或者很低的残余误差平方和，这就是你需要寻找的最佳平衡点

## 过拟合

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565194296032.rc6pawbsfj.png)

过拟合给回归造成的影响
过拟合效果很差的原因

过拟合在训练集中表现得很好，但在测试集中表现很差，方差很高，泛化效果不好

## 带有特征数量的平衡误差

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.bhsj3kstjw.png)

找到最高点的过程，正则化。

## 正则化

正则化是自动处理模型中使用的额外特征的方式。

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/1565226669368.iue8d6ceu7f.png)

一般的回归是要最大程度地降低拟合中的误差平方和。缩短拟合与任何指定数据点之间的距离或者距离平方。
Lasso Regression 在最小化误差平方和的同时，还要最小化使用的特征数量  
β 描述的就是使用的特征数量  
这个公式精确地规定了更少的误差和使用更少特征数量更简单的拟合之间的平衡  
所以 Lasso Regression 自动考虑惩罚参数 λ，这样它就能帮你指出哪些特征对你的回归有最重要的影响，在发现这些特征后，它就能减少或删除用处不大的特征的系数。


![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.z3qtax96dfa.png)

```
import sklearn.linear_model.Lasso
features,labels = GetMyData()
regression = Lasso()
regression.fit(features,labels)
regression.predict([2,4]) 
print regression.coef_  #将返回回归找到的所有系数的列表
```



![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.abw5vkte83.png)



# Mini-project

过拟合算法的一种传统方式是使用大量特征和少量训练数据。你可以在 `feature_selection/find_signature.py` 中找到初始代码。准备好决策树，开始在训练数据上进行训练，打印出准确率。

###  根据初始代码，有多少训练点？
`150`

###  你刚才创建的决策树的准确率是多少？

（记住，我们设置决策树用于过拟合——理想情况下，我们希望看到的是相对较低的测试准确率。）

```
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

clf = DecisionTreeClassifier()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
# accuracy_score method 1
score = clf.score(features_test, labels_test)
# accuracy_score method 2
acc = accuracy_score(pred, labels_test)
print(acc)
```

```
0.95
```
### 识别最大特征
选择（过拟合）决策树并使用 `feature_importances_` 属性来获得一个列表， 其中列出了所有用到的特征的相对重要性（由于是文本数据，因此列表会很长）。 我们建议迭代此列表并且仅在超过阈值（比如 0.2——**记住，所有单词都同等重要，每个单词的重要性都低于 0.01**）的情况下将特征重要性打印出来。

- 最重要特征的重要性是什么？该特征的数字是多少？

```
# 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大
feature_importance = clf.feature_importances_
print(f"特征重要性最大值：{max(feature_importance)}, 特征重要性最小值：{min(feature_importance)}")
print(f"该特征单词的数量：{feature_importance.argmax()}")
```

```
特征重要性最大值：0.764705882353 , 特征重要性最小值：0.0
该特征单词的数量：33614
```

方法2：

```
# 所有单词都同等重要，每个单词的重要性都低于 0.01
imp_list = []
for index, feature in enumerate(feature_importance):
    if feature > 0.2:
        imp_list.append(feature)
        print(f"单词词索引：{index}, 单词词数量：{feature}, 单词：{words_bag[index]}")
if not imp_list:
    print("无超过阈值0.2的特征重要性")
```



### 使用Tfldf获得最重要的单词

为了确定是什么单词导致了问题的发生，你需要返回至 TfIdf，使用你从迷你项目的上一部分中获得的特征数量来获取关联词。 你可以在 TfIdf 中调用 get_feature_names() 来返回包含所有单词的列表； 抽出造成大多数决策树歧视的单词。

- 这个单词是什么？类似于签名这种与 Chris Germany 或 Sara Shackleton 唯一关联的单词是否讲得通？

```
# 使用 TfIdf 获得最重要的单词
print(f"获得对应的单词是：{words_bag[feature_importance.argmax()]}")  #sshacklensf
```

### 删除、重复

从某种意义上说，这一单词看起来像是一个异常值，所以让我们在删除它之后重新拟合。 返回至 `text_learning/vectorize_text.py`，使用我们删除“sara”、“chris”等的方法，从邮件中删除此单词。 重新运行 `vectorize_text.py`，完成以后立即重新运行 `find_signature.py`。

- 有跳出其他任何的异常值吗？是什么单词？像是一个签名类型的单词？（跟之前一样，将异常值定义为重要性大于 0.2 的特征）。

```
text = text.replace("sara", "").replace("shackleton", '').replace("chris", '') \
                .replace("germani", '').replace("sshacklensf", '')
```

再次更新 `vectorize_test.py` 后重新运行。然后，再次运行 `find_signature.py`。

- 是否出现其他任何的重要特征（重要性大于 0.2）？有多少？它们看起来像“签名文字”，还是更像来自邮件正文的“邮件内容文字”？

```
text = text.replace("sara", "").replace("shackleton", '').replace("chris", '') \
     .replace("germani", '').replace("sshacklensf", '').replace("cgermannsf",'')
```

### 再次检查重要特征

再次更新 `vectorize_test.py` 后重新运行。然后，再次运行 `find_signature.py`。

- 是否出现其他任何的重要特征（重要性大于 0.2）？有多少？它们看起来像“签名文字”，还是更像来自邮件正文的“邮件内容文字”？

是的，还有一个词("houectect")。你对这个词的意思的猜测和我们的一样好，但是它看起来不像一个明显的签名词，所以让我们继续前进，不要去掉它。

### 过拟合树的准确率

- 现在决策树的准确率是多少？

```
0.816837315131
```

我们已经移除了两个“签名词语”，所以要让我们的算法拟合训练集，同时不出现过拟合更为困难。 记住，我们这里是想要知道我们是否会让算法过拟合，准确率如何并不是关键！



以上数据为原课程的准确答案，本人在此Mini-project中，代码运行完全不同，过拟合决策树准确率为1.0，最大特征重要性一直为0。

原因分析：可能数据集文件发生了变化，词干化的结果与原来课程的结果发生很大的变化。

`vectorize_test.py`检测可以得到正确的结果。



## SelectKBest

[Sklearn SelectKBest](<https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>)

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.7ugjkcfroys.png)

- score_func 
- [`f_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif)

  ANOVA F-value between label/feature for classification tasks.

- [`mutual_info_classif`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif)

  Mutual information for a discrete target.

- [`chi2`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2)

  Chi-squared stats of non-negative features for classification tasks.

- [`f_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_regression.html#sklearn.feature_selection.f_regression)

  F-value between label/feature for regression tasks.

- [`mutual_info_regression`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_regression.html#sklearn.feature_selection.mutual_info_regression)

  Mutual information for a continuous target.

- [`SelectPercentile`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectPercentile.html#sklearn.feature_selection.SelectPercentile)

  Select features based on percentile of the highest scores.

- [`SelectFpr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFpr.html#sklearn.feature_selection.SelectFpr)

  Select features based on a false positive rate test.

- [`SelectFdr`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFdr.html#sklearn.feature_selection.SelectFdr)

  Select features based on an estimated false discovery rate.

- [`SelectFwe`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectFwe.html#sklearn.feature_selection.SelectFwe)

  Select features based on family-wise error rate.

- [`GenericUnivariateSelect`](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.GenericUnivariateSelect.html#sklearn.feature_selection.GenericUnivariateSelect)

  Univariate feature selector with configurable mode.

Examples:

```
>>> from sklearn.datasets import load_digits
>>> from sklearn.feature_selection import SelectKBest, chi2
>>> X, y = load_digits(return_X_y=True)
>>> X.shape
(1797, 64)
>>> X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
>>> X_new.shape
(1797, 20)
```

