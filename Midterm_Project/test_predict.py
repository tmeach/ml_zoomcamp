#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'

customer = {
    "BMI": 30.0,
    "Smoking": 'Yes',
    "AlcoholDrinking": 'Yes',
    "Stroke": "No",
    "PhysicalHealth": 10.0,
    "MentalHealth": 15.0,
    "DiffWalking": 'Yes',
    "Sex": 'Male',
    "AgeCategory": '75-79',
    "Race": 'White',
    "Diabetic": 'Yes',
    "PhysicalActivity": 'No',
    "GenHealth": 'Good',
    "SleepTime": 4.0,
    "Asthma": 'Yes',
    "KidneyDisease": 'Yes',
    "SkinCancer": 'No'
}


response = requests.post(url, json=customer).json()
print(response) 

if response['get_disease'] == True:
    print('sending cautionary email to %s' % customer_id)
else:
    print('not sending cautionary email to  %s' % customer_id)
