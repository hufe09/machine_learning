#!/usr/bin/python

import sys
import pickle

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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


features_list_nopoi = email_data + payment_data + stock_data + new_features
features_list = ['poi'] + email_data + payment_data + stock_data + new_features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

initial_features_list = ['poi'] + payment_data + stock_data + email_data
# Create a dataframe from the dictionary for manipulation
df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)
df = df[initial_features_list]

# Fill in the missing financial data with zeros
df[payment_data] = df[payment_data].fillna(value=0)
df[stock_data] = df[stock_data].fillna(value=0)

# Fill in the missing email meta data with the mean for poi or nonpoi
# imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

df_poi = df[df['poi'] == True]
df_nopoi = df[df['poi'] == False]

df_poi_copy = df_poi.copy()
df_nopoi_copy = df_nopoi.copy()
df_poi_copy.loc[:, email_data] = imp.fit_transform(df_poi.loc[:, email_data])
df_nopoi_copy.loc[:, email_data] = imp.fit_transform(df_nopoi.loc[:, email_data])
df = df_poi_copy.append(df_nopoi_copy)


# Drop the identified outliers
df.drop(axis=0, labels=['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'], inplace=True)
df.drop(axis=0, labels=['FREVERT MARK A', 'LAVORATO JOHN J', 'WHALLEY LAWRENCE G', 'BAXTER JOHN C'], inplace=True)

# Add in additional features to dataframe
df['to_poi_ratio'] = df['from_poi_to_this_person'] / df['to_messages']
df['from_poi_ratio'] = df['from_this_person_to_poi'] / df['from_messages']
df['shared_poi_ratio'] = df['shared_receipt_with_poi'] / df['to_messages']
df['bonus_to_salary'] = df['bonus'] / df['salary']
df['bonus_to_total'] = df['bonus'] / df['total_payments']
df.fillna(value=0, inplace=True)

# Scale the data frame
scaled_df = df.copy()
scaled_df.loc[:, features_list_nopoi] = scale(scaled_df.loc[:, features_list_nopoi])

# Create my_dataset
my_dataset = scaled_df.to_dict(orient='index')

clf = Pipeline([
    ('select_features', SelectKBest(k=19)),
    ('classify', DecisionTreeClassifier(criterion='entropy', max_depth=None, max_features=None, min_samples_split=20))
])

dump_classifier_and_data(clf, my_dataset, features_list)
