# 识别来自安然电子邮件的欺诈行为

## 项目概述

安然曾是 2000 年美国最大的公司之一。2002 年，由于其存在大量的企业欺诈行为，这个昔日的大集团土崩瓦解。 在随后联邦进行的调查过程中，大量有代表性的保密信息进入了公众的视线，包括成千上万涉及高管的邮件和详细的财务数据。 你将在此项目中扮演侦探，运用你的新技能，根据安然丑闻中公开的财务和邮件数据来构建相关人士识别符。 为了协助你进行侦查工作，我们已将数据与手动整理出来的欺诈案涉案人员列表进行了合并， 这意味着被起诉的人员要么达成和解，要么向政府签署认罪协议，再或者出庭作证以获得免受起诉的豁免权。

## 需要的资源

相关文件如下所示：

`poi_id.py`：用于 POI 识别符的初始代码，你将在此处撰写你的分析报告。

`final_project_dataset.pkl`：项目数据集，详情如下。

`tester.py`：在你提交供优达学城评估的分析报告时，你将随附算法、数据集和你使用的特征列表（这些是在 `poi_id.py` 中自动创建的）。 评估人员将在此后使用这一代码来测试你的结果，以确保性能与你在报告中所述类似。你无需处理这一代码，我们只是将它呈现出来供你参考。

emails_by_address：该目录包含许多文本文件，每个文件又包含特定邮箱的往来邮件。 你可以进行参考，并且可以根据邮件数据集的详细信息创建更多的高级特征。你无需处理电子邮件语料库来完成项目。

## 迈向成功

我们将给予你可读入数据的初始代码，将你选择的特征放入 numpy 数组中，该数组是大多数 sklearn 函数假定的输入表单。 你要做的就是设计特征，选择并调整算法，用以测试和评估识别符。 我们在设计数个迷你项目之初就想到了这个最终的项目，因此请记得借助你已完成的工作成果。

在预处理此项目时，我们已将安然邮件和财务数据与字典结合在一起，字典中的每对键值对应一个人。 字典键是人名，值是另一个字典（包含此人的所有特征名和对应的值）。 数据中的特征分为三大类，即财务特征、邮件特征和 POI 标签。

**财务特征**: [`salary`,  `deferral_payments`, `total_payments`, `loan_advances`, `bonus`, `restricted_stock_deferred`, `deferred_income`, `total_stock_value`, `expenses`, `exercised_stock_options`, `other`, `long_term_incentive`, `restricted_stock`, `director_fees`] (单位均是美元）

**邮件特征**: [`to_messages`, `email_address`, `from_poi_to_this_person`, `from_messages`, `from_this_person_to_poi`, `shared_receipt_with_poi`] (单位通常是电子邮件的数量，明显的例外是 `email_address`，这是一个字符串）

**POI 标签**: [ `poi` ] (boolean，整数)

我们鼓励你在启动器功能中制作，转换或重新调整新功能。如果这样做，你应该把新功能存储到 my_dataset，如果你想在最终算法中使用新功能，你还应该将功能名称添加到 my_feature_list，以便于你的评估者可以在测试期间访问它。关于如何在数据集中添加具体的新要素的例子，可以参考“特征选择”这一课。

# Task 1. 加载数据集
## 数据预处理

- 将原字典转为DataFrame

```
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)
df = df[initial_features_list]
```

- 用0代替空值

```
df[payment_data] = df[payment_data].fillna(value=0)
df[stock_data] = df[stock_data].fillna(value=0)
```

- 以均值填充缺失值

```
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

df_poi = df[df['poi'] == True]
df_nonpoi = df[df['poi'] == False]

df_poi_copy = df_poi.copy()
df_nonpoi_copy = df_nonpoi.copy()
df_poi_copy.loc[:, email_data] = imp.fit_transform(df_poi.loc[:, email_data])
df_nonpoi_copy.loc[:, email_data] = imp.fit_transform(df_nonpoi.loc[:, email_data])
df = df_poi_copy.append(df_nonpoi_copy)
```
## 移除异常值
```
# Drop the identified outliers
df.drop(axis=0, labels=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
df.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C'], inplace=True)
```

# Task 2. 特征选择与新增

## 选择特征

将所有**财务特征**汇总为`payment_data`,  **股票特征**汇总为 `stock_data`,  **邮件特征**汇总为`email_data`。

```
payment_data = ['salary',
                'bonus',
                'long_term_incentive',
                'deferred_income',
                'deferral_payments',
                'loan_advances',
                'other',
                'expenses',
                'director_fees',
                'total_payments']

stock_data = ['exercised_stock_options',
              'restricted_stock',
              'restricted_stock_deferred',
              'total_stock_value']

email_data = ['to_messages',
              'from_messages',
              'from_poi_to_this_person',
              'from_this_person_to_poi',
              'shared_receipt_with_poi']
```

## 添加新特征

- `to_poi_ratio` POI发给此人的邮件占所有邮件比例
- `from_poi_ratio` 此人发给POI的邮件占所有邮件比例
- `bonus_to_salary` 奖金占工资比例
- `bonus_to_total` 奖金占所有收入比例
- 用0填充空值

```
new_features = ['to_poi_ratio',
                'from_poi_ratio',
                'shared_poi_ratio',
                'bonus_to_salary',
                'bonus_to_total']
```

```
df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments']
df.fillna(value=0, inplace=True)
```
## 特征列表整合
```
initial_features_list = ['poi'] + payment_data + stock_data + email_data
features_list_nonpoi = email_data + payment_data + stock_data + new_features
features_list = ['poi'] + email_data + payment_data + stock_data + new_features
```

# Task 3. 尝试各种算法
## Gaussian Naive Bayes

```
from sklearn.naive_bayes import GaussianNB

clf_naive_bayes = GaussianNB()
param_grid_naive_bayes = {
    'var_smoothing': [1e-3, 1e-6, 1e-9]
}
```
## Decision Tree
```
from sklearn.tree import DecisionTreeClassifier

clf_decision_tree = DecisionTreeClassifier()
param_grid_decision_tree = {
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5]
}
```
## SVM
```
from sklearn.svm import SVC

clf_svm = SVC(kernel='rbf', class_weight='balanced')
param_grid_svm = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.001, 0.005, 0.01, 0.1],
}
```
## KNN

```
from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier()
```
## Random Forest
```
from sklearn.ensemble import RandomForestClassifier

clf_random_forest = RandomForestClassifier()
param_grid_random_forest = {
    "criterion": ["entropy", 'gini'],
    'n_estimators': [5, 10, 15],  # 默认为10
}
```

# Task 4. 特征降维

## PCA 

[sklearn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

### 所有特征
```
# initial_features_list = ['poi'] + payment_data + stock_data + email_data

features_list = initial_features_list
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)
```
```
from sklearn.decomposition import PCA

n_components = 9 # 取值为9时，F1值最高

pca = PCA(n_components=n_components, whiten=True).fit(features)
print(pca)
print("PCA可释方差", pca.explained_variance_ratio_)
print("方差量", pca.explained_variance_)
features_pca = pca.transform(features)
```

```
PCA可释方差 [0.6539102  0.16044838 0.05412837 0.04283136 0.0273554  0.024117
 0.01814221 0.00570084 0.00411326 0.00308973]
方差量 [12.50997867  3.06954346  1.03553181  0.81940819  0.52333706  0.4613832
  0.34707929  0.10906301  0.07869083  0.05910982]
```
- 测试算法为 SVC
```
from sklearn.svm import SVC

clf_svm = SVC(kernel='rbf', class_weight='balanced')
param_grid_svm = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.001, 0.005, 0.01, 0.1],
}
```
```
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.97      0.90      0.94        41
         POI       0.33      0.67      0.44         3
    accuracy                           0.89        44
   macro avg       0.65      0.78      0.69        44
weighted avg       0.93      0.89      0.90        44
```
### 金融特征

```
features_list = ['poi'] + payment_data
n_components = len(payment_data)
```

```
PCA可释方差 [9.27479946e-01 4.92239755e-02 9.37762579e-03 5.48690406e-03
 3.34152533e-03 2.31228311e-03 1.43628927e-03 8.78149279e-04
 3.14773477e-04 1.48528298e-04]
方差量 [9.33876359e+00 4.95634512e-01 9.44229907e-02 5.52474478e-02
 3.36457034e-02 2.32822989e-02 1.44619471e-02 8.84205481e-03
 3.16944329e-03 1.49552631e-03]
```

```
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       1.00      0.73      0.85        41
         POI       0.21      1.00      0.35         3
    accuracy                           0.75        44
   macro avg       0.61      0.87      0.60        44
weighted avg       0.95      0.75      0.81        44
```

### 股票特征

```
features_list = ['poi'] + stock_data
n_components = len(stock_data)
```
```
PCA可释方差 [8.11840330e-01 1.84598095e-01 3.55726188e-03 4.31337231e-06]
方差量 [3.26975691e+00 7.43484742e-01 1.43271789e-02 1.73724788e-05]
```

```
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.95      0.98      0.96        41
         POI       0.50      0.33      0.40         3
    accuracy                           0.93        44
   macro avg       0.73      0.65      0.68        44
weighted avg       0.92      0.93      0.93        44
```

### 邮件特征

```
features_list = ['poi'] + email_data
n_components = 1
```
```
PCA可释方差 [0.6084821]
方差量 [3.06339264]
```

```
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.97      0.80      0.88        41
         POI       0.20      0.67      0.31         3
    accuracy                           0.80        44
   macro avg       0.59      0.74      0.59        44
weighted avg       0.92      0.80      0.84        44
```

### 可视化 PCA 特征

```
features_train_pca = [features_train[x][0] for x in range(len(features_train))]
features_test_pca = [features_test[x][0] for x in range(len(features_test))]
features_pca_x = features_train_pca + features_test_pca

def get_max_index(data_list):
    max_idx = 0
    x_max = max(data_list)
    for idx, value in enumerate(data_list):
        if value == x_max:
            max_idx = idx
    return max_idx


for i in range(4):
    # 移除噪点
    features_pca_x.pop(get_max_index(features_pca_x))
    i += 1

plt.scatter([x for x in range(len(features_pca_x))], features_pca_x)
plt.show()
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.yglzcqzcsi9.png)



## 寻找最大特征

```python
import pprint
def find_max_imp(feature_importance):
    """
    寻找最大重要性特征
    :param feature_importance: clf.feature_importances_  特征最大重要性
    :return: 元组列表， [(imp_value，index)]
    """
    res = []
    for index, value in enumerate(feature_importance):
        res.append((value, index))
    res.sort(key=lambda x: x[0], reverse=True)
    return res


feature_importance = None
try:
    feature_importance = clf.feature_importances_
    if len(feature_importance) > 0:
        feature_importance = find_max_imp(feature_importance)
        print("按特征重要性排序")
        importance_list = [(features_list[feature_importance[i][1]], feature_importance[i][0]) for i in
                           range(len(feature_importance))]
        pprint.pprint(importance_list)
        print("重要性最大的特征对应数据")
        pprint.pprint(remove_nan(features_list[feature_importance[0][1]]))
except Exception as e:
    print(e)
```


- Decision Tree

```
按特征重要性排序
[('loan_advances', 0.48340167463084954),
 ('from_messages', 0.1475178635168029),
 ('long_term_incentive', 0.1253506646104134),
 ('total_stock_value', 0.0878143392188336),
 ('deferred_income', 0.08498849173096579),
 ('from_this_person_to_poi', 0.07092696629213481),
 ('poi', 0.0),
...
]
重要性最大的特征对应数据
[('TOTAL', 83925000),
 ('LAY KENNETH L', 81525000),
 ('FREVERT MARK A', 2000000),
 ('PICKERING MARK R', 400000)]
```

- Random Forest
```
# 1.
按特征重要性排序
[('loan_advances', 0.12742406292279015),
 ('from_poi_to_this_person', 0.11784832841837897),
 ('deferred_income', 0.10382931246233489),
 ('poi', 0.08989106547858272),
 ('deferral_payments', 0.08075506379989161),
 ('from_messages', 0.06871766195498545),
 ('salary', 0.06071114648540047),
 ('from_this_person_to_poi', 0.0559018989288589),
 ('total_stock_value', 0.05425558890243085),
 ('exercised_stock_options', 0.05290413760965681),
 ('shared_receipt_with_poi', 0.04784000127817178),
 ('restricted_stock_deferred', 0.04134719700647645),
 ('expenses', 0.04072525660883523),
 ('other', 0.033726746976225956),
 ('long_term_incentive', 0.01716301350978771),
 ('to_messages', 0.006959517657192075),
 ('total_payments', 0.0),
 ...
 ]
 重要性最大的特征对应数据
[('TOTAL', 83925000),
 ('LAY KENNETH L', 81525000),
 ('FREVERT MARK A', 2000000),
 ('PICKERING MARK R', 400000)]
```
```
# 2.
按特征重要性排序
[('from_poi_to_this_person', 0.12558741176111904),
 ('from_this_person_to_poi', 0.12139336182847475),
 ('shared_receipt_with_poi', 0.10427778571364081),
 ('from_messages', 0.09146072079816654),
 ('salary', 0.08743779164849837),
 ('restricted_stock_deferred', 0.07614220994828882),
 ('loan_advances', 0.0662344357695773),
 ('deferred_income', 0.06390843229036972),
 ('poi', 0.05463457196764156),
 ('long_term_incentive', 0.04423995075976428),
 ('total_stock_value', 0.043972890838527554),
 ('expenses', 0.03387936444895538),
 ('deferral_payments', 0.030276138345140452),
 ('exercised_stock_options', 0.017302327302327305),
 ('other', 0.01427497346654955),
 ('to_messages', 0.013137403897729305),
 ('restricted_stock', 0.008161616161616161),
 ('total_payments', 0.0036786130536130557),
 ('bonus', 0.0)]
重要性最大的特征对应数据
[('LAVORATO JOHN J', 528),
 ('DIETRICH JANET R', 305),
 ('KITCHEN LOUISE', 251),
 ('FREVERT MARK A', 242),
 ...
 ]
```
```
# 3.
按特征重要性排序
[('loan_advances', 0.16804129823307265),
 ('from_poi_to_this_person', 0.13673203716799412),
 ('deferred_income', 0.1291346226744345),
 ('restricted_stock_deferred', 0.10858430354997263),
 ('salary', 0.05889914563619017),
 ('total_stock_value', 0.05071828836725237),
 ('shared_receipt_with_poi', 0.04794541521841269),
 ('expenses', 0.04735835767475849),
 ('long_term_incentive', 0.04384315472746463),
 ('exercised_stock_options', 0.03592997198879552),
 ('from_messages', 0.033753252833309016),
 ('total_payments', 0.03037593984962407),
 ('deferral_payments', 0.029950092190239985),
 ('from_this_person_to_poi', 0.02770798455800099),
 ('poi', 0.025988672527278735),
 ('to_messages', 0.018459433824602366),
 ('other', 0.00657802897859715),
 ('bonus', 0.0),
 ('restricted_stock', 0.0)]
重要性最大的特征对应数据
[('TOTAL', 83925000),
 ('LAY KENNETH L', 81525000),
 ('FREVERT MARK A', 2000000),
 ('PICKERING MARK R', 400000)]
```

>Random forest 获得的特征重要性会发生变化，不固定。

```
AttributeError: 'GaussianNB' object has no attribute 'feature_importances_'
'SVC' object has no attribute 'feature_importances_'
'KNeighborsClassifier' object has no attribute 'feature_importances_'
```
>Naive bayes, SVM, KNN 没有 `feature_importances_`属性

### 可视化重要性最大的特征

`clf=clf_random_forest`

```
def visual_features(data, x_axis, y_axis):
    for point in data:
        x_data = point[x_axis]
        y_data = point[y_axis]
        plt.scatter(x_data, y_data)

    x_label = features_list[x_axis]
    y_label = features_list[y_axis]
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig(f"./out_images/{y_label}-{x_label}.png")
    plt.show()
    
# 可视化特征（线性）
# X 轴为salary(features_list[1]) Y轴为重要性最大的特征
visual_features(data, 1, feature_importance[0][1])
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.9muovauua7n.png)

# Task 5. 移除异常值

根据之前的最大重要性特征数据以及，绘制的图像，可以看出有一个很明显的**异常值**，将其剔除。

所有数据否指向了 "TOTAL"，剔除他。这一步在**异常值**课程中已经遇到过。执行相同操作即可。

```
# 从df源数据中剔除 异常值
df.drop(axis=0, labels=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
df.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C'], inplace=True)
```

```
# 从data_dict源数据中剔除 异常值
outliers_list = ['TOTAL', 'SKILLING JEFFREY K']
for name in outliers_list:
    data_dict.pop(name)
```

最重要特征为 `from_messages`时，可以发现，最高发邮件数为 14000多封。
```
Score: 0.8372093023255814
Predicting time: 0.003 s
按特征重要性排序
[('from_messages', 0.15158442217051138),...]
```
![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.nsxvr2k2lkd.png)

最重要特征为`total_stock_value`时

```
Score: 0.9069767441860465
Predicting time: 0.004 s
按特征重要性排序
[('total_stock_value', 0.12291636618871937),...]
```

![image](https://raw.githubusercontent.com/hufe09/GitNote-Images/master/Picee/image.mfcxdf6q73a.png)


# 选择并调整算法
## SelectKBest

[sklearn SelectKBest](<https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SelectKBest.html>)

```
from sklearn.feature_selection import SelectKBest, f_classif

features_new = SelectKBest(f_classif, k="all").fit_transform(features, labels)
print(features_new[0])
```
```
[ 2.019550e+05  2.902000e+03  2.869717e+06  4.484442e+06  0.000000e+00
  4.175000e+06 -1.260270e+05 -3.081055e+06  1.729541e+06  1.386800e+04
  4.700000e+01  1.729541e+06  2.195000e+03  1.520000e+02  6.500000e+01
  3.048050e+05  1.407000e+03  1.260270e+05  0.000000e+00]
```

# Task 6. 性能测试

## 交叉验证

```
features_list = ['poi'] + email_data + payment_data + stock_data + new_features
# PCA 
n_components = 3
```

### SVM

```
from sklearn.model_selection import GridSearchCV

print("Fitting the classifier to the training set")

clf_svm = SVC(kernel='rbf', class_weight='balanced')
param_grid_svm = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.001, 0.005, 0.01, 0.1],
}

clf = GridSearchCV(clf_svm, param_grid_svm, cv=3, n_jobs=1)

t0 = time()
clf.fit(features_train, labels_train)
print("Fitting time:", round(time() - t0, 3), "s")

t1 = time()
print("Predicting the features on the testing set")
pred = clf.predict(features_test)
print('Score:', clf.score(features_test, labels_test))
print("Predicting time:", round(time() - t1, 3), "s")

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

print("构建显示主要分类指标的文本报告")
target_names = ["NO-POI", "POI"]
print(classification_report(labels_test, pred, target_names=target_names))
print("计算混淆矩阵以评估分类的准确性")
print(confusion_matrix(labels_test, pred, labels=list([0, 1])))
```

```
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight='balanced',
                           coef0=0.0, decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=1,
             param_grid={'C': [1000.0, 5000.0, 10000.0, 50000.0, 100000.0],
                         'gamma': [0.001, 0.005, 0.01, 0.1]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.179 s
Predicting the features on the testing set
Score: 0.7142857142857143
Predicting time: 0.0 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.93      0.72      0.81        36
         POI       0.29      0.67      0.40         6
    accuracy                           0.71        42
   macro avg       0.61      0.69      0.61        42
weighted avg       0.84      0.71      0.75        42
计算混淆矩阵以评估分类的准确性
[[26 10]
 [ 2  4]]
```

### Gaussian Naive Bayes

```
clf_naive_bayes = GaussianNB()
param_grid_naive_bayes = {
    'var_smoothing': [1e-3, 1e-6, 1e-9]
}

my_clf, my_param_grid = clf_naive_bayes, param_grid_naive_bayes

clf = GridSearchCV(my_clf, my_param_grid, cv=3, n_jobs=1)
```

```
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=GaussianNB(priors=None, var_smoothing=1e-09), iid='warn',
             n_jobs=1, param_grid={'var_smoothing': [0.001, 1e-06, 1e-09]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.019 s
Predicting the features on the testing set
Score: 0.8809523809523809
Predicting time: 0.001 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.90      0.97      0.93        36
         POI       0.67      0.33      0.44         6
    accuracy                           0.88        42
   macro avg       0.78      0.65      0.69        42
weighted avg       0.86      0.88      0.86        42
计算混淆矩阵以评估分类的准确性
[[35  1]
 [ 4  2]]
```

### Decision Tree

```
clf_decision_tree = DecisionTreeClassifier()
param_grid_decision_tree = {
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5]
}

my_clf, my_param_grid = clf_decision_tree, param_grid_decision_tree
```

```
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=DecisionTreeClassifier(class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features=None,
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              presort=False, random_state=None,
                                              splitter='best'),
             iid='warn', n_jobs=1,
             param_grid={'criterion': ['gini', 'entropy'],
                         'min_samples_leaf': [1, 3, 5]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.041 s
Predicting the features on the testing set
Score: 0.8333333333333334
Predicting time: 0.0 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.91      0.89      0.90        36
         POI       0.43      0.50      0.46         6
    accuracy                           0.83        42
   macro avg       0.67      0.69      0.68        42
weighted avg       0.84      0.83      0.84        42
计算混淆矩阵以评估分类的准确性
[[32  4]
 [ 3  3]]
```
### KNN

```
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=KNeighborsClassifier(algorithm='auto', leaf_size=30,
                                            metric='minkowski',
                                            metric_params=None, n_jobs=None,
                                            n_neighbors=5, p=2,
                                            weights='uniform'),
             iid='warn', n_jobs=1, param_grid={}, pre_dispatch='2*n_jobs',
             refit=True, return_train_score=False, scoring=None, verbose=0)
Fitting time: 0.015 s
Predicting the features on the testing set
Score: 0.8333333333333334
Predicting time: 0.015 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.87      0.94      0.91        36
         POI       0.33      0.17      0.22         6
    accuracy                           0.83        42
   macro avg       0.60      0.56      0.56        42
weighted avg       0.79      0.83      0.81        42
计算混淆矩阵以评估分类的准确性
[[34  2]
 [ 5  1]]
```

### Random Forest

```
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=RandomForestClassifier(bootstrap=True, class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features='auto',
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              n_estimators='warn', n_jobs=None,
                                              oob_score=False,
                                              random_state=None, verbose=0,
                                              warm_start=False),
             iid='warn', n_jobs=1,
             param_grid={'criterion': ['entropy'], 'n_estimators': [10]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.098 s
Predicting the features on the testing set
Score: 0.8571428571428571
Predicting time: 0.003 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.88      0.97      0.92        36
         POI       0.50      0.17      0.25         6
    accuracy                           0.86        42
   macro avg       0.69      0.57      0.59        42
weighted avg       0.82      0.86      0.83        42
计算混淆矩阵以评估分类的准确性
[[35  1]
 [ 5  1]]
```

> 随机森林算法结果不稳定

对比Gaussian Naive Bayes、SVM、Decision Tree、KNN和 Random Forest 的结果，发现Random Forest 算法F1值较高，但不够稳定，最终算法使用 SVM 和 Decision Tree 进行验证。（也可以选择其他）

# Task 7.  最终测试

执行 `grid.fit()` 函数时

（1）先执行特征预处理 `MinMaxScaler()` 的 `fit`  和  `transform` 函数，将执行后的结果传递给下一个参数，即 PCA

（2）上一步的数据继续执行特征降维函数` PCA()` 或 `SelectKBest()`的 `fit`  和 `transform`  函数，生成结果传递给下一步，即分类器

（3）最后执行 `scm.SVC()` 等分类器。

```
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([
    ("process", MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('svm', clf_svm),
    # ("decision_tree", clf_decision_tree),
])

N_FEATURES_OPTIONS = [3]

param_grid = [
    {
        'reduce_dim': [PCA()],
        'reduce_dim__n_components': N_FEATURES_OPTIONS,
        
        'svm__C': param_grid_svm["C"],
        'svm__gamma': param_grid_svm["gamma"],

        # 'decision_tree__criterion': param_grid_decision_tree['criterion'],
        # 'decision_tree__min_samples_leaf': param_grid_decision_tree['min_samples_leaf'],
    },
]

grid = GridSearchCV(pipe, cv=3, n_jobs=1, param_grid=param_grid)

my_classifier(grid, my_dataset, features_list, folds=1000)
```

Out:

```
The size of enron dataset: (146, 20)
********************** Compute a PCA  on the dataset ******************************
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
PCA可释方差 [0.51804182 0.14641958 0.09735164]
方差量 [12.51874851  3.53830497  2.35255269]
****************************** Tune your classifier ******************************
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight='balanced',
                           coef0=0.0, decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=1,
             param_grid={'C': [1000.0, 5000.0, 10000.0, 50000.0, 100000.0],
                         'gamma': [0.001, 0.005, 0.01, 0.1]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.227 s
Predicting the features on the testing set
Score: 0.7727272727272727
Predicting time: 0.0 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.94      0.80      0.87        41
         POI       0.11      0.33      0.17         3
    accuracy                           0.77        44
   macro avg       0.53      0.57      0.52        44
weighted avg       0.89      0.77      0.82        44
计算混淆矩阵以评估分类的准确性
[[33  8]
 [ 2  1]]
****************************** Test your classifier ******************************
Start test ...
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=Pipeline(memory=None,
                                steps=[('process',
                                        MinMaxScaler(copy=True,
                                                     feature_range=(0, 1))),
                                       ('reduce_dim',
                                        PCA(copy=True, iterated_power='auto',
                                            n_components=None,
                                            random_state=None,
                                            svd_solver='auto', tol=0.0,
                                            whiten=False)),
                                       ('svm',
                                        SVC(C=1.0, cache_size=200,
                                            class_weight='balanced', coef0=0.0,
                                            decision_func...
             iid='warn', n_jobs=1,
             param_grid=[{'reduce_dim': [PCA(copy=True, iterated_power='auto',
                                             n_components=3, random_state=None,
                                             svd_solver='auto', tol=0.0,
                                             whiten=False)],
                          'reduce_dim__n_components': [3],
                          'svm__C': [1000.0, 5000.0, 10000.0, 50000.0,
                                     100000.0],
                          'svm__gamma': [0.001, 0.005, 0.01, 0.1]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
{'false_negatives': 357,
 'false_positives': 2152,
 'total_predictions': 15000,
 'true_negatives': 10848,
 'true_positives': 1643}
{'F1': 0.5670405522001726,
 'F2': 0.6964815599830437,
 'accuracy': 0.8327333333333333,
 'precision': 0.4329380764163373,
 'recall': 0.8215}
The test was completed. It took 384.633 s
```
- N_FEATURES_OPTIONS = [1, 3, 9]

```
The size of enron dataset: (146, 20)
************************* Compute a PCA  on the dataset ******************************
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
PCA可释方差 [0.2700528  0.14036819 0.10788011]
方差量 [6.52789512 3.39307269 2.6077493 ]
****************************** Tune your classifier ******************************
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=SVC(C=1.0, cache_size=200, class_weight='balanced',
                           coef0=0.0, decision_function_shape='ovr', degree=3,
                           gamma='auto_deprecated', kernel='rbf', max_iter=-1,
                           probability=False, random_state=None, shrinking=True,
                           tol=0.001, verbose=False),
             iid='warn', n_jobs=1,
             param_grid={'C': [1000.0, 5000.0, 10000.0, 50000.0, 100000.0],
                         'gamma': [0.001, 0.005, 0.01, 0.1]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.146 s
Predicting the features on the testing set
Score: 0.7142857142857143
Predicting time: 0.001 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.93      0.72      0.81        36
         POI       0.29      0.67      0.40         6
    accuracy                           0.71        42
   macro avg       0.61      0.69      0.61        42
weighted avg       0.84      0.71      0.75        42
计算混淆矩阵以评估分类的准确性
[[26 10]
 [ 2  4]]
****************************** Test your classifier ******************************
Start test ...
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=Pipeline(memory=None,
                                steps=[('process',
                                        MinMaxScaler(copy=True,
                                                     feature_range=(0, 1))),
                                       ('reduce_dim',
                                        PCA(copy=True, iterated_power='auto',
                                            n_components=None,
                                            random_state=None,
                                            svd_solver='auto', tol=0.0,
                                            whiten=False)),
                                       ('svm',
                                        SVC(C=1.0, cache_size=200,
                                            class_weight='balanced', coef0=0.0,
                                            decision_func...
             iid='warn', n_jobs=1,
             param_grid=[{'reduce_dim': [PCA(copy=True, iterated_power='auto',
                                             n_components=3, random_state=None,
                                             svd_solver='auto', tol=0.0,
                                             whiten=False)],
                          'reduce_dim__n_components': [1, 3, 9],
                          'svm__C': [1000.0, 5000.0, 10000.0, 50000.0,
                                     100000.0],
                          'svm__gamma': [0.001, 0.005, 0.01, 0.1]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
{'false_negatives': 696,
 'false_positives': 1660,
 'total_predictions': 14000,
 'true_negatives': 10340,
 'true_positives': 1304}
{'F1': 0.5253827558420628,
 'F2': 0.5946734768332725,
 'accuracy': 0.8317142857142857,
 'precision': 0.4399460188933873,
 'recall': 0.652}
The test was completed. It took 1035.459 s
```

- 决策树

```
The size of enron dataset: (146, 20)
************************ Compute a PCA  on the dataset ******************************
PCA(copy=True, iterated_power='auto', n_components=3, random_state=None,
    svd_solver='auto', tol=0.0, whiten=False)
PCA可释方差 [0.2700528  0.14036819 0.10788011]
方差量 [6.52789512 3.39307269 2.6077493 ]
****************************** Tune your classifier ******************************
Fitting the classifier to the training set
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=DecisionTreeClassifier(class_weight=None,
                                              criterion='gini', max_depth=None,
                                              max_features=None,
                                              max_leaf_nodes=None,
                                              min_impurity_decrease=0.0,
                                              min_impurity_split=None,
                                              min_samples_leaf=1,
                                              min_samples_split=2,
                                              min_weight_fraction_leaf=0.0,
                                              presort=False, random_state=None,
                                              splitter='best'),
             iid='warn', n_jobs=1,
             param_grid={'criterion': ['gini', 'entropy'],
                         'min_samples_leaf': [1, 3, 5]},
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
Fitting time: 0.049 s
Predicting the features on the testing set
Score: 0.8333333333333334
Predicting time: 0.0 s
构建显示主要分类指标的文本报告
              precision    recall  f1-score   support
      NO-POI       0.91      0.89      0.90        36
         POI       0.43      0.50      0.46         6
    accuracy                           0.83        42
   macro avg       0.67      0.69      0.68        42
weighted avg       0.84      0.83      0.84        42
计算混淆矩阵以评估分类的准确性
[[32  4]
 [ 3  3]]
****************************** Test your classifier ******************************
Start test ...
GridSearchCV(cv=3, error_score='raise-deprecating',
             estimator=Pipeline(memory=None,
                                steps=[('process',
                                        MinMaxScaler(copy=True,
                                                     feature_range=(0, 1))),
                                       ('reduce_dim',
                                        PCA(copy=True, iterated_power='auto',
                                            n_components=None,
                                            random_state=None,
                                            svd_solver='auto', tol=0.0,
                                            whiten=False)),
                                       ('decision_tree',
                                        DecisionTreeClassifier(class_weight=None,
                                                               criterion='gini',
                                                               max_...
             iid='warn', n_jobs=1,
             param_grid=[{'decision_tree__criterion': ['gini', 'entropy'],
                          'decision_tree__min_samples_leaf': [1, 3, 5],
                          'reduce_dim': [PCA(copy=True, iterated_power='auto',
                                             n_components=1, random_state=None,
                                             svd_solver='auto', tol=0.0,
                                             whiten=False)],
                          'reduce_dim__n_components': [1, 3, 9]}],
             pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
             scoring=None, verbose=0)
{'false_negatives': 1157,
 'false_positives': 885,
 'total_predictions': 14000,
 'true_negatives': 11115,
 'true_positives': 843}
{'F1': 0.4522532188841202,
 'F2': 0.43328536184210525,
 'accuracy': 0.8541428571428571,
 'precision': 0.4878472222222222,
 'recall': 0.4215}
The test was completed. It took 180.65 s
```

