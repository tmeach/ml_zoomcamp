## Midterm Project - A prediction of Heart Disease

![Alt text](image.png)

### Overview

I'm going to apply everything I've learned so far into practice. This is end-to-end ML project including model training and deployment. 

In general the project includes the following steps: 
- Description of the project and explanation of how a model could be used. 
- Preparing data, EDA and analyzing important features.
- Training several models, tune their performance and selecting the best one.
- Puting the model into a web service and deploy it locally with Docker.
- As a bonus deploying it to the cloud

### Project Description
I'm going to work on the Heart Disease prediction project. It's a binary classification task. 

The project "A prediction of Heart Disease" is based on a dataset that contains data from the Centers for Disease Control and Prevention (CDC) in the USA. It represents the results of an annual survey of over 400,000 adults concerning their health and lifestyle. Heart diseases are one of the leading causes of death, and the project aims to identify the factors that influence the likelihood of developing these diseases.

The dataset includes information about factors such as high blood pressure, high cholesterol levels, smoking, diabetes, obesity (high body mass index), insufficient physical activity, and excessive alcohol consumption. 

The primary objective of this research is to determine the factors that have a substantial influence on the likelihood of heart diseases in order to take some preventive measures by a specific person.

Data: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease

Load the data:
```bash
wget https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
```

## Step-by-step guide

### 1. Clone this repo:
```bash
git clone https://github.com/tmeach/project-of-the-week.git
```

### 2. Dependencies: 
   
Note that I've used pipenv environment. In the repo you can find Pipfile and Pipfile.lock files. In order to create virtual environment and install all dependecies please do the following steps: 

Install pipenv environment
```
pip install pipenv
```
Create virtual environment (pipenv) with dependencies from Pipfile
```
pipenv install <name_your_environment>
```
Activate pipenv environment
```
pipenv shell 
```

### 3. Docker:
   
Note that in repo you can see Dockerfile and you can run application is Docker container: 
```bash
#run docker deamon 
dockerd/sudo dockerd

#create docker image
docker build -t <your_image_name> .

#create docker container and run it 
docker run -it --rm -p 9696:9696 <your_image_name>
```
