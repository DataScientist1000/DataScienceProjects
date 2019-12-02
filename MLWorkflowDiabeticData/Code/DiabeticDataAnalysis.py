# -*- coding: utf-8 -*-
#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
path = os.path.abspath("MLWorkflowDiabeticData/Data/diabetes.csv")
"""
#Data Exploration
"""
data = pd.read_csv(path)
data.columns
data.info()

data.Outcome.value_counts()
data.groupby('Outcome').size()

data.head()

print("Diabetic Dataset dimension : {}".format(data.shape))

data.groupby('Outcome').hist(figsize=(9,9))

#------------------------------------------------------------
"""
#-- Data Cleaning
"""
#------------------------------------------------------------
#Missing Null values
data.isnull().sum()
data.isna().sum()

#outliers

print("Total :",data[data.BloodPressure == 0].shape[0])
#print("Total :",data[data.BloodPressure == 0].shape[1])

print(data[data.BloodPressure == 0].groupby('Outcome')['Age'].count())

data.BloodPressure.value_counts()
data[data.BloodPressure == 0].value_counts()

print("Total : ", data[data.Glucose == 0].shape[0])

print(data[data.Glucose == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.SkinThickness == 0].shape[0])

print(data[data.SkinThickness == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.BMI == 0].shape[0])

print(data[data.BMI == 0].groupby('Outcome')['Age'].count())

print("Total : ", data[data.Insulin == 0].shape[0])
print(data[data.Insulin == 0].groupby('Outcome')['Age'].count())

#--------------------------------------------------------
#“BloodPressure”, “BMI” and “Glucose” are zero.
#--------------------------------------------------------
data_mod = data[(data.BloodPressure != 0) & (data.Glucose != 0) & (data.BMI != 0)]

print(data_mod .shape)

#--------------------------------------------------------
"""
# Feature Engineering
"""
#--------------------------------------------------------

feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = data_mod[feature_names]
y = data_mod.Outcome

#--------------------------------------------------------
"""
#Model Selection
K-Nearest Neighbors, 
Support Vector Classifier, 
Logistic Regression, 
Gaussian Naive Bayes, 
Random Forest and Gradient Boost 
"""
#--------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

models = []

models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('NB', GaussianNB()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('GBC', GradientBoostingClassifier()))

#--------------------------------------------------------
#Evaluation Methods
#1. Train/Test Fold
#2. K-Fold Cross Validation
#--------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = data_mod.Outcome, random_state=0)

names = []
scores = []

for name , model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores.append(accuracy_score( y_test, y_pred ))
    names.append(name)
    
tr_split = pd.DataFrame({'Name':names, 'Score': scores})
    
print(tr_split)

#K-Fold Cross Validation
K_names = []
K_scores = []

strat_k_fold =  StratifiedKFold(n_splits =10, random_state=10)

for name , model in models:
    kfold = KFold(n_splits = 10, random_state = 10)
    score = cross_val_score(model,X , y , cv= kfold, scoring = 'accuracy').mean()
    K_names.append(name)
    K_scores.append(score)
    
kf_cross_val = pd.DataFrame({'Name' : K_names, 'Score': K_scores})
print(kf_cross_val)

#plot the accuracies
axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()

#--------------------------------------------------------
"""
Feature Engineering
Logistic Regression — Feature Selection
"""
#--------------------------------------------------------
from sklearn.feature_selection import RFECV

logreg_model = LogisticRegression()
rfecv = RFECV(estimator=logreg_model, step=1, cv=strat_k_fold, scoring='accuracy')
rfecv.fit(X, y)

plt.figure()
plt.title('Logistic Regression CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


feature_importance = list(zip(feature_names, rfecv.support_))
new_features = []
for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
        
print(new_features)

# Calculate accuracy scores 
X_new = data_mod[new_features]

initial_score = cross_val_score(logreg_model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))

fe_score = cross_val_score(logreg_model, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))
#--------------------------------------------------------
#Gradient Boosting — Feature Selection
#--------------------------------------------------------
gb_model = GradientBoostingClassifier()
gb_rfecv = RFECV(estimator=gb_model, step=1, cv=strat_k_fold, scoring='accuracy')
gb_rfecv.fit(X, y)
plt.figure()
plt.title('Gradient Boost CV score vs No of Features')
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(gb_rfecv.grid_scores_) + 1), gb_rfecv.grid_scores_)
plt.show()

feature_importance = list(zip(feature_names, gb_rfecv.support_))
new_features = []
for key,value in enumerate(feature_importance):
    if(value[1]) == True:
        new_features.append(value[0])
        
print(new_features)

X_new_gb = data_mod[new_features]

initial_score = cross_val_score(gb_model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Initial accuracy : {} ".format(initial_score))

fe_score = cross_val_score(gb_model, X_new_gb, y, cv=strat_k_fold, scoring='accuracy').mean()
print("Accuracy after Feature Selection : {} ".format(fe_score))

#--------------------------------------------------------
#Model Parameter Tuning
#--------------------------------------------------------
from sklearn.model_selection import GridSearchCV

# Specify parameters
c_values = list(np.arange(1, 10))
param_grid = [
    {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
    {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
]

grid = GridSearchCV(LogisticRegression(), param_grid, cv=strat_k_fold, scoring='accuracy')
grid.fit(X_new, y)

print(grid.best_params_)
print(grid.best_estimator_)

logreg_new = LogisticRegression(C=1, multi_class='ovr', penalty='l2', solver='liblinear')

initial_score = cross_val_score(logreg_new, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()

print("Final accuracy : {} ".format(initial_score))

#END