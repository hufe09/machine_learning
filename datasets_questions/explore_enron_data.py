#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))
print(enron_data)


print(f"安然数据集大小: {len(enron_data)}")


person_names = [person_name for person_name in enron_data]
person_names.sort()
features = [feature for feature in enron_data[person_names[0]]]
print(f"安然数据集中特征数量: {len(features)}")
print("安然数据集中特征")
print(features)


poi_names = []
for person_name in person_names:
    if enron_data[person_name]["poi"] == 1:
        poi_names.append(person_name)
print(f"安然数据集中嫌疑人数量: {len(poi_names)}")
print("嫌疑人(POI)数量及名单")
print(poi_names)

# poi_names.txt中嫌疑人数量
with open("../final_project/poi_names.txt", "r") as file:
    poi_names_txt = file.readlines()[2:]
    print(f"poi_names.txt中嫌疑人数量为:{len(poi_names_txt)}人")
    poi_dict = {}
    n_count, y_count = 0, 0
    for poi in poi_names_txt:
        poi_dict[poi[4:-2]] = poi[1]
        if poi[1] == "n":
            n_count += 1
        elif poi[1] == "y":
            y_count += 1
print(f"poi_names.txt中嫌疑人中安然的员工有{y_count}人")
print(poi_dict)

print(f"James Prentice 名下的股票总值是多少？ {enron_data['PRENTICE JAMES']['total_stock_value']}")
print(f"我们有多少来自 Wesley Colwell 的发给嫌疑人的电子邮件？ {enron_data['COLWELL WESLEY']['from_this_person_to_poi']}")
print(f"Jeffrey Skilling 行使的股票期权价值是多少？ {enron_data['SKILLING JEFFREY K']['exercised_stock_options']}")


print(f"安然的 CEO Jeffrey K. Skilling 拿了多少钱？{enron_data['SKILLING JEFFREY K']['total_payments']}")
print(f"安然的 董事会主席 Kenneth L. Lay 拿了多少钱？{enron_data['LAY KENNETH L']['total_payments']}")
print(f"安然的 CFO（首席财务官） Andrew S. Fastow 拿了多少钱？{enron_data['FASTOW ANDREW S']['total_payments']}")

no_salary, no_email = [], []
for person_name in person_names:
    if enron_data[person_name]["salary"] != "NaN":
        no_salary.append(person_name)
    if enron_data[person_name]["email_address"] != "NaN":
        no_email.append(person_name)

print(f"此数据集中有多少雇员有量化的工资？{len(no_salary)}人有")
print(f"此数据集中已知的邮箱地址是否可用？{len(no_email)}人可用")


def static_nan(data_set, feature, names=person_names):
    nans = []
    for name in names:
        if data_set[name][feature] == "NaN":
            nans.append(name)
    return f"{feature}被设置为`NaN`的数量为{len(nans)}，比例为{round(len(nans) / len(names), 2)}"


print("整个数据集中", static_nan(enron_data, "total_payments"))
print("POI中", static_nan(enron_data, "total_payments", poi_names))
