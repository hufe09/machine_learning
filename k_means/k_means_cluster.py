#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    # plot each cluster with a different color--add more colors for
    # drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color=colors[pred[ii]])

    # if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.title(f"{feature_2}-{feature_1}-{len(features_list) - 1}features")
    plt.savefig(name)
    plt.show()


# load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
# there's an outlier--remove it!
data_dict.pop("TOTAL", 0)

# the input features we want to use
# can be any key in the person-level dictionary (salary, director_fees, etc.)
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi = "poi"
# features_list = [poi, feature_1, feature_2]
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

# in the "clustering with 3 features" part of the mini-project,
# you'll want to change this line to
# for f1, f2, _ in finance_features:
# (as it's currently written, the line below assumes 2 features)
# for f1, f2 in finance_features:
for f1, f2, _ in finance_features:
    plt.scatter(f1, f2)
plt.savefig("features_scatter.png")
plt.show()

# cluster here; create predictions of the cluster labels
# for the data and store them to a list called pred

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=0).fit(finance_features)
pred = kmeans.predict(finance_features)

print("Maximum exercised_stock_options: ", max(f2 for f1, f2, f3 in finance_features))
print("Minimum exercised_stock_options: ", min(f2 for f1, f2, f3 in finance_features if f2 != 0))
print("Maximum salary: ", max(f1 for f1, f2, f3 in finance_features))
print("Minimum salary: ", min(f1 for f1, f2, f3 in finance_features if f1 != 0))

# rename the "name" parameter when you change the number of features
# so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name=f"clusters-{len(features_list) - 1}features.png",
         f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named pred found, no clusters to plot")

# 特征缩放
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
min_max_x = min_max_scaler.fit_transform(finance_features)
print(f"每个功能相对缩放数据 {min_max_scaler.scale_}")
print(f"[[200000.0, 1000000.0, 1061827.0]] 特征缩放值：{min_max_scaler.transform([[200000.0, 1000000.0, 1061827.0]])}")
print(f"20万的 `salary` 特征缩放值：{200000.0 * min_max_scaler.scale_[0]}")
print(f"100万的 `exercised_stock_options` 特征缩放值：：{1000000.0 * min_max_scaler.scale_[1]}")
