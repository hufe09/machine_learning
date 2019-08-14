#!/usr/bin/python

from time import time
import sys
import pickle
import pprint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import warnings

warnings.filterwarnings('ignore')


def my_classifier(clf, dataset, feature_list, folds=1000):
    print("Start test ...")
    t0 = time()

    data = featureFormat(dataset, feature_list, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(n_splits=folds, random_state=42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv.split(features, labels):
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        # fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    model_score = {}
    try:
        pprint.pprint(clf)
        static_sict = {}

        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        static_sict["total_predictions"] = total_predictions
        static_sict["true_negatives"] = true_negatives
        static_sict["false_negatives"] = false_negatives
        static_sict["false_positives"] = false_positives
        static_sict["true_positives"] = true_positives

        pprint.pprint(static_sict)

        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        model_score["accuracy"] = accuracy
        precision = 1.0 * true_positives / (true_positives + false_positives)
        model_score["precision"] = precision
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        model_score["recall"] = recall
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        model_score["F1"] = f1
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        model_score["F2"] = f2
    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")
    finally:
        if model_score:
            pprint.pprint(model_score)
    print("The test was completed. It took", round(time() - t0, 3), "s")


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


# Access enron dataset
# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

###########################################################################
# Task 1: Select what features you'll use.
# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

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

new_features = ['to_poi_ratio',
                'from_poi_ratio',
                'shared_poi_ratio',
                'bonus_to_salary',
                'bonus_to_total']

initial_features_list = ['poi'] + payment_data + stock_data + email_data
features_list_nonpoi = email_data + payment_data + stock_data + new_features
features_list = ['poi'] + email_data + payment_data + stock_data + new_features

# features_list = initial_features_list
# features_list = ['poi'] + payment_data
# features_list = ['poi'] + stock_data
# features_list = ['poi'] + email_data

# Create a dataframe from the dictionary for manipulation
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)
df = df[initial_features_list]

# Fill in the missing financial data with zeros
df[payment_data] = df[payment_data].fillna(value=0)
df[stock_data] = df[stock_data].fillna(value=0)

# Fill in the missing email meta data with the mean for poi or nonpoi
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy='mean')

df_poi = df[df['poi'] == True]
df_nonpoi = df[df['poi'] == False]

df_poi_copy = df_poi.copy()
df_nonpoi_copy = df_nonpoi.copy()
df_poi_copy.loc[:, email_data] = imp.fit_transform(df_poi.loc[:, email_data])
df_nonpoi_copy.loc[:, email_data] = imp.fit_transform(df_nonpoi.loc[:, email_data])
df = df_poi_copy.append(df_nonpoi_copy)
print("The size of enron dataset:", df.shape)

###########################################################################
# Task 2: Remove outliers
# Drop the identified outliers
df.drop(axis=0, labels=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
df.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C'], inplace=True)

# Task 3: Create new feature(s)
# Add in additional features to dataframe
df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments']
df.fillna(value=0, inplace=True)

# Scale the data frame
from sklearn.preprocessing import scale

scaled_df = df.copy()
scaled_df.loc[:, features_list_nonpoi] = scale(scaled_df.loc[:, features_list_nonpoi])

# Create my_dataset
my_dataset = scaled_df.to_dict(orient='index')

# Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

######################################################################
print("*" * 30, "Compute a PCA  on the dataset", "*" * 30)
# Compute a PCA  on the dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction

from sklearn.decomposition import PCA


def pca_reduce_dim(n_components):
    pca = PCA(n_components=n_components).fit(features)
    print(pca)
    print("PCA可释方差", pca.explained_variance_ratio_)
    print("方差量", pca.explained_variance_)
    return pca.transform(features_train), pca.transform(features_test)


n_components = [9, len(payment_data), len(stock_data), len(email_data)][0]
n_components = [1, 3, 9, 15, 24][1]
features_train, features_test = pca_reduce_dim(n_components)

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

###########################################################################
# Task 4: Try a varity of classifiers
# Please name your classifier clf for easy export below.
# Note that if you want to do PCA or other multi-stage operations,
# you'll need to use Pipelines. For more info:
# http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB

clf_naive_bayes = GaussianNB()
param_grid_naive_bayes = {
    'var_smoothing': [1e-3, 1e-6, 1e-9]
}

from sklearn.tree import DecisionTreeClassifier

clf_decision_tree = DecisionTreeClassifier()
param_grid_decision_tree = {
    "criterion": ["gini", "entropy"],
    "min_samples_leaf": [1, 3, 5]
}

from sklearn.svm import SVC

clf_svm = SVC(kernel='rbf', class_weight='balanced')
param_grid_svm = {
    'C': [1e3, 5e3, 1e4, 5e4, 1e5],
    'gamma': [0.001, 0.005, 0.01, 0.1],
}

from sklearn.neighbors import KNeighborsClassifier

clf_knn = KNeighborsClassifier()

from sklearn.ensemble import RandomForestClassifier

clf_random_forest = RandomForestClassifier()
param_grid_random_forest = {
    "criterion": ["entropy", 'gini'],
    'n_estimators': [5, 10, 15],  # 默认为10
}

###########################################################################
print("*" * 30, "Tune your classifier", "*" * 30)
# Task 5: Tune your classifier to achieve better than .3 precision and recall
# using our testing script. Check the tester.py script in the final project
# folder for details on the evaluation method, especially the test_classifier
# function. Because of the small size of the dataset, the script uses
# stratified shuffle split cross validation. For more info:
# http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import GridSearchCV

print("Fitting the classifier to the training set")

# my_clf, my_param_grid = clf_naive_bayes, param_grid_naive_bayes
# my_clf, my_param_grid = clf_svm, param_grid_svm
my_clf, my_param_grid = clf_decision_tree, param_grid_decision_tree
# my_clf, my_param_grid = clf_knn, {}
# my_clf, my_param_grid = clf_random_forest, param_grid_random_forest

clf = GridSearchCV(my_clf, my_param_grid, cv=3, n_jobs=1)
print(clf)
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
################################################################################
# Validate and Evaluate
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler

pipe = Pipeline([
    ("process", MinMaxScaler()),
    ('reduce_dim', PCA()),
    ('svm', clf_svm),
    # ("decision_tree", clf_decision_tree),
])

N_FEATURES_OPTIONS = [1, 3, 9, ]

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

# Task 6: Dump your classifier, dataset, and features_list so anyone can
# check your results. You do not need to change anything below, but make sure
# that the version of poi_id.py that you submit can be run on its own and
# generates the necessary .pkl files for validating your results.

dump_classifier_and_data(grid, my_dataset, features_list)

print("*" * 30, "Test your classifier", "*" * 30)
my_classifier(grid, my_dataset, features_list, folds=1000)
