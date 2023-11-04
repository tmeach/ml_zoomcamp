#!/usr/bin/env python
# coding: utf-8


import pandas as pd 
import numpy as np 


from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import joblib
# Data preparation

df = pd.read_csv('/home/timur/work_hub/ml_zoomcamp2023/Midterm_Project/heart_2020_cleaned.csv')
df = df.drop_duplicates()
df = df[(df['BMI'] < 43.03) & (3.0<=df['SleepTime']) & (df['SleepTime']<=11.0)]


df[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']] = df[['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Asthma', 'KidneyDisease', 'SkinCancer']].replace({'Yes':1, 'No':0})
df['Sex'] = df['Sex'].replace({'Male':1, 'Female':0})

df.dtypes[df.dtypes == 'object']
object_columns = list(df.dtypes[df.dtypes == 'object'].index)
for i in object_columns:
    df[i] = df[i].str.lower().str.replace(' ', '_')
    
# Split dataset    
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# Set target
y_full_train = df_full_train['HeartDisease']
y_train = df_train['HeartDisease']
y_val = df_val['HeartDisease']
y_test = df_test['HeartDisease']

del df_full_train['HeartDisease']
del df_train['HeartDisease']
del df_val['HeartDisease']
del df_test['HeartDisease']    

# train model

dv = DictVectorizer(sparse=False)

dicts_full_train = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=dv.get_feature_names_out())
dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out())


xgb_params = {
    'eta': 0.05,
    'max_depth': 3,
    'min_child_weight':20,
    
    'objective':'binary:logistic',
    'eval_metric':'auc',
    
    'nthread': 8,
    'seed': 1,
    'verbosity':1
}

model = xgb.train(xgb_params,
                  dfulltrain,
                  num_boost_round=175)

y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print(f'auc={auc}')


# save the model
output_file = "model.model"
model.save_model(output_file)

# save the dv
dv_output_file = "dv.pkl"
joblib.dump(dv, dv_output_file)
