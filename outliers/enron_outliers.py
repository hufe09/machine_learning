#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

# read in data dictionary, convert to numpy array
data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
features = ["salary", "bonus"]
# data_dict.pop("TOTAL")
data = featureFormat(data_dict, features)

# your code below
for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter(salary, bonus)

plt.xlabel("salary")
plt.ylabel("bonus")
plt.savefig("bonus-salary-outliers.png")
# plt.savefig("bonus-salary-outliers-remove_TOTAL.png")
plt.show()


def remove_nan(feature):
    feature_no_nan = []
    for i, item in data_dict.items():
        if item[feature] != "NaN":
            feature_no_nan.append((i, item[feature]))
    feature_no_nan.sort(key=lambda x: x[1], reverse=True)
    return feature_no_nan


if __name__ == "__main__":
    print(remove_nan("bonus"))
