Udacity Machine Learning 
==============

优达学城[机器学习入门](<https://classroom.udacity.com/courses/ud120>)课程项目记录。课程 Github地址 [ud120-projects](<https://github.com/udacity/ud120-projects>)

# 机器学习入门

- 第01课：欢迎来到机器学习，认识 Sebastian 和 Katie，和他们一起讨论机器学习。
- 第02课：[朴素贝叶斯](./naive_bayes)，学习分类，训练和测试，并使用 scikit-learn 运行一个朴素贝叶斯分类器。
- 第03课：[支持向量机（SVM）](./svm)，了解支持向量机（SVM）的原理，并使用scikit-learn 学习编写一个SVM。
- 课04课：[决策树](./decision_tree)，学习决策树算法的工作原理，包括熵和信息增益。
- 第05课：[选择自己的算法](./choose_your_own)，在这个迷你项目中，您将通过选择自己的算法来扩展算法工具箱，对地形数据进行分类，包括K-Means，AdaBoost 和 Random Forest。对比各算法的性能。
- 第06课：[数据集与问题](./datasets_questions)，开始了解和 Mini 项目中使用安然数据集。
- 第07课：[回归](./regression)，学习如何使用线性回归为连续数据建模。学习评估回归的两种指标 R 平方和最小绝对误差。
- 第08课：[异常值](./outliers)，讨论如何选择和移除异常值。
- 第09课：[聚类](./k_means)，讨论无监督学习的，并了解如何使用 scikit-learn 的 K-Means 算法。
- 第10课：[特征缩放](./k_means)，学习特征缩放，了解哪些算法在使用前需要首先进行特征缩放。
- 第11课：[文本学习](./text_learning)，学习如何在机器学习算法中使用文本数据。
- 第12课：[特征选择](./feature_selection)，讨论在什么时候以及为什么使用特征选择，并提供了一些实现此目的的技巧。学习正则化、偏差、方差。使用单变量特征选择工具SelectKBest 选择k个最高分特征。
- 第13课：[主成分分析（PCA）](./pca)，通过主成分分析（PCA）学习数据维度和减少维度数量。分析Scikit-learn 示例中的人脸识别项目。
- 第14课：[交叉验证](./validation)，了解有关测试，训练，交叉验证和参数网格搜索。学习Sklearn 中的 GridSearchCV 进行K折交叉验证。使用 PIpline 按顺序应用变换列表和最终估算器。
- 第15课：[评估指标](./evaluation)，如何知道我们的分类器是否表现良好？分类器的不同评估指标。
- 第16课：[最终项目](./final_project)，在此项目中，将扮演侦探，运用机器学习技能构建一个算法，通过公开的安然财务和邮件数据集，找出有欺诈嫌疑的安然雇员。

## 项目准备

1. 检查你是否装有可用的 python，版本最好是 3.7，课程原来 Github 项目使用 Python 版本为 2.7，本项目所有代码均将使用 3.7 版本替换，scikit-learn 版本 0.21.
2. 我们会使用 pip 来安装一些包。首先，从[此处](https://pip.pypa.io/en/latest/installing.html)获取并安装 pip。
3. 使用 pip 安装一系列 Python 包：
   - 转到终端行界面（请勿打开 Python，只打开命令提示符）
   - 安装 sklearn: **pip install scikit-learn**
   - [此处](http://scikit-learn.org/stable/install.html)包含 sklearn 安装说明，可供参考
4. 安装自然语言工具包：**pip install nltk**

你只需操作一次，基础代码包含所有迷你项目的初始代码。进入 **tools/** 目录，运行 **startup.py**。该程序首先检查 python 模块，然后下载并解压缩我们在后期将大量使用的大型数据集：**安然数据集**。下载和解压缩需要一些时间，但是你无需等到全部完成再开始第一部分。

## 安然数据集

安然欺诈案是一个混乱而又引人入胜的大事件，从中可以发现几乎所有想像得到的企业违法行为。安然的电子邮件和财务数据集还是巨大、混乱的信息宝藏，而且，在你稍微熟悉这些宝藏后，它们会变得更加有用。我们已将这些电子邮件和财务数据合并为一个数据集，而你将在此迷你项目中研究它。

POI (Person Of Interest) 指的是在这个问题中的相关人物。

建议你在有时间的时候，花一个半小时，看一下【纪录片】安然：房间里最聪明的人 Enron: The Smartest Guys in the Room ( [Youtube链接](https://www.youtube.com/watch?v=rDyMz1V-GSg) )来了解这个事件。

**下载数据集**

目前从[CMU 安然数据集页面](https://www.cs.cmu.edu/~./enron/) 能下载到的最新数据集为 **[May 7, 2015 Version of dataset](https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz)**.

## 错误

[错误记录](./tools/errors_solution.md)

