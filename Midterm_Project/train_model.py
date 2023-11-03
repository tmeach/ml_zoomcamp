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


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# Set target
y_train = df_train['HeartDisease']
y_val = df_val['HeartDisease']
y_test = df_test['HeartDisease']

del df_train['HeartDisease']
del df_val['HeartDisease']
del df_test['HeartDisease']    

# train model

train_dict = df_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)


dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=dv.get_feature_names_out())
dval = xgb.DMatrix(X_val, label=y_val, feature_names=dv.get_feature_names_out())

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
                  dtrain,
                  num_boost_round=175)

y_pred = model.predict(dval)
auc = roc_auc_score(y_val, y_pred)
print(f'auc={auc}')


# save the model
output_file = "model.model"
model.save_model(output_file)

# save the dv
dv_output_file = "dv.pkl"
joblib.dump(dv, dv_output_file)












# def train(df_train, y_train, C=1.0):
    
#     dicts = df_train[numerical + categorical].to_dict(orient='records')
    
#     dv = DictVectorizer(sparse=False)
#     X_train = dv.fit_transform(dicts)
    
#     model = LogisticRegression(C=C, max_iter=1000)
#     model.fit(X_train, y_train)
    
#     return dv, model


# def predict(df, dv, model):
#     dicts = df[numerical + categorical].to_dict(orient='records')
    
#     X = dv.transform(dicts)
#     y_pred = model.predict_proba(X)[:, 1]
    
#     return y_pred

# print(f'doing validation with C={C}')
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

# scores = []

# fold=0
# for train_idx, val_idx in kfold.split(df_full_train):
#     df_train = df_full_train.iloc[train_idx]
#     df_val = df_full_train.iloc[val_idx]

#     y_train = df_train.churn.values
#     y_val = df_val.churn.values

#     dv, model = train(df_train, y_train)
#     y_pred = predict(df_val, dv, model)

#     auc = roc_auc_score(y_val, y_pred)
#     scores.append(auc)
    
#     print(f'auc on fold {fold} is {auc}')
#     fold = fold + 1

# print('validation results:')
# print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# print('training the final model')

# dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
# y_pred = predict(df_test, dv, model)

# y_test = df_test.churn.values
# auc = roc_auc_score(y_test, y_pred)

# print(f'auc={auc}')

# with open(output_file, 'wb') as f_out:
#     pickle.dump((dv, model), f_out)
    
# print(f'the model is saved to {output_file}')    