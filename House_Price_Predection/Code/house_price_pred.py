# -*- coding: utf-8 -*-
"""
Import all the libraries
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
import os

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category = DeprecationWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

path = os.path.abspath("House_Price_Predection/Data/train.csv")
path_test = os.path.abspath("House_Price_Predection/Data/test.csv")

train_data = pd.read_csv(path) 
test_data = pd.read_csv(path_test)

"""
Loading and Analysisng the data
"""

train_data.head()
train_data.columns

train_data.describe()

train_data.shape

test_data.shape

train_data.keys()

correlations = train_data.corr()

correlations = correlations['SalePrice'].sort_values(ascending = False)

features = correlations.index[1:6]

"""
Imputing NUll Values

For numerical imputing, we would typically fill the missing values with a measure like median, mean, or mode. 
For categorical imputing, I chose to fill the missing values with the most common term that appeared from the entire column
"""

train_null = pd.isnull(train_data).sum()
test_null = pd.isnull(test_data).sum()


null = pd.concat([train_null,test_null], axis =1, keys=['Train', 'Test'])

null_many = null[null.sum(axis=1) > 200]
null_few = null[(null.sum(axis=1) > 0 ) & (null.sum(axis=1) < 200 )]

#you can find these features on the description data file provided

null_has_meaning = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", 
                    "GarageType", "GarageFinish", "GarageQual", "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for i in null_has_meaning:
    train_data[i].fillna("None", inplace= True)
    test_data[i].fillna("None", inplace= True)
    
"""
Imputing "Real" NaN Values
"""
from sklearn.preprocessing  import Imputer

imputer = Imputer(strategy='median')

train_null = pd.isnull(train_data).sum()
test_null = pd.isnull(test_data).sum()
null = pd.concat([train_null,test_null], axis =1, keys=['Train', 'Test'])
null_many = null[null.sum(axis=1) > 200]
null_few = null[(null.sum(axis=1) > 0 ) & (null.sum(axis=1) < 200 )]

null_many
#LotFrontage has lot of values it is better to drop them.

train_data.drop('LotFrontage', axis= 1, inplace = True)
test_data.drop('LotFrontage', axis= 1, inplace = True)

null_few
"""
GarageYrBlt, MasVnrArea, and MasVnrType all have a fairly decent amount of missing values. 
MasVnrType is categorical so we can replace the missing values with "None", as we did before. 
We can fill the others with median.
"""
train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].median(), inplace = True)
test_data['GarageYrBlt'].fillna(test_data['GarageYrBlt'].median(), inplace = True)
train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].median(), inplace = True)
test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].median(), inplace = True)
train_data['MasVnrType'].fillna("None", inplace = True)
test_data['MasVnrType'].fillna("None", inplace = True)

#Let's move on to the features with fewer missing values.
types_train = train_data.dtypes
num_train_data = types_train[(types_train == "int64") | (types_train == "float64") ]
cat_train_data = types_train[(types_train == "object")]

types_test = test_data.dtypes
num_test_data = types_test[(types_test == "int64") | (types_test == "float64") ]
cat_test_data = types_test[(types_test == "object")]

"""
Numerical Imputing
We'll impute with median since the distributions are probably very skewed. 
we should convert num_train and num_test to a list to make it easier to work with

"""
numerical_values_train = list(num_train_data.index)
numerical_values_test = list(num_test_data.index)

fill_num = []
for i in numerical_values_train:
    if i in list(null_few.index):
        fill_num.append(i)

#These are the numerical features in the data that have missing values in them. We will impute these features with a for-loop below.
print(fill_num)

for i in fill_num:
    train_data[i].fillna(train_data[i].median(), inplace =True)
    test_data[i].fillna(test_data[i].median(),inplace = True)
    

"""
Categorical Imputing

Since these are categorical values, we can't impute with median or mean. We can, however, use mode.
We'll impute with the most common term that appears in the entire list.
"""

categorical_values_train = list(cat_train_data.index)
categorical_values_test = list(cat_test_data.index)

#These are all the categorical features in our data
print(categorical_values_train)

fill_cat = []

for i in categorical_values_train:
  if i in list(null_few.index):
      fill_cat.append(i)

#These are the categorical features that have missing values in them. We'll impute with the most common term.

print(fill_cat)

def most_common_term(lst):
    lst = list(lst)
    return max(set(lst),key=lst.count)

#most_common_term finds the most common term in a series

most_common = ["Electrical", "Exterior1st", "Exterior2nd", "Functional", "KitchenQual", "MSZoning", 
               "SaleType", "Utilities", "MasVnrType"]
    

counter = 0

for i in fill_cat:
    most_common[counter] = most_common_term(train_data[i])
    counter += 1 
    
most_common_dict  = {fill_cat[0]: [most_common[0]], 
                     fill_cat[1]: [most_common[1]], 
                     fill_cat[2]: [most_common[2]], 
                     fill_cat[3]: [most_common[3]],
                     fill_cat[4]: [most_common[4]], 
                     fill_cat[5]: [most_common[5]], 
                     fill_cat[6]: [most_common[6]], 
                     fill_cat[7]: [most_common[7]],
                     fill_cat[8]: [most_common[8]]}

most_common_dict

#Replace the null categorical filed with these values
counter=0

for i in fill_cat:
    train_data[i].fillna(most_common[counter], inplace = True)
    test_data[i].fillna(most_common[counter], inplace = True)
    counter += 1
    
#Final check for null values after appling the logic
train_null = pd.isnull(train_data).sum()
test_null = pd.isnull(test_data).sum()

null_check = pd.concat([train_null, test_null], axis = 1, keys=['Train', 'Test'] )

null_check[null_check.sum(axis=1) > 0]

"""
Feature Engineering:
We need to create feature vectors in order to get the data ready to be fed into our model as training data.
This requires us to convert the categorical values into representative numbers.
"""    

sns.distplot(train_data["SalePrice"])

sns.distplot(np.log(train_data['SalePrice']))

"""
It appears that the target, SalePrice, is very skewed and a transformation like a logarithm would make it more normally distributed. 
Machine Learning models tend to work much better with normally distributed targets, rather than greatly skewed targets.
 By transforming the prices, we can boost model performance.
"""
train_data['TransformedPrice'] = np.log(train_data['SalePrice'])

#let's take a look at all the categorical features in the data that need to be transformed.

categorical_values_train = list(cat_train_data.index)
categorical_values_test = list(cat_test_data.index)
print(categorical_values_train)
print(categorical_values_test)

for i in categorical_values_train:
    feature_set = set(train_data[i])
    for j in feature_set:
        feature_list = list(feature_set)
        train_data.loc[train_data[i] == j, i] = feature_list.index(j)

for i in categorical_values_test:
    feature_set2 = set(test_data[i])
    for j in feature_set2:
        feature_list2 = list(feature_set2)
        test_data.loc[test_data[i] == j, i] = feature_list2.index(j)
        
train_data.head()
        
"""
Creating, Training, Evaluating, Validating, and Testing ML Models
In classification, we used accuracy as a evaluation metric; in regression, we will use the R^2 score as well as the RMSE to evaluate our model performance. 
We will also use cross validation to optimize our model hyperparameters.
"""
from sklearn.linear_model import LinearRegression,Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score,KFold

X_train = train_data.drop(['Id', 'SalePrice','TransformedPrice'], axis=1 ).values

y_train = train_data['TransformedPrice'].values

X_test = train_data.drop("Id", axis= 1).values

#spliiting the dataset
from sklearn.model_selection import train_test_split

X_training, X_valid, y_training, y_valid = train_test_split(X_train,y_train, test_size = 0.2, random_state = 0)

"""
Linear Regression Model
"""
linreg = LinearRegression()
parameters_lin = {"fit_intercept" : [True, False], "normalize" : [True, False], "copy_X" : [True, False]}
grid_linreg = GridSearchCV(linreg, parameters_lin, verbose=1 , scoring = "r2")
grid_linreg.fit(X_training, y_training)

print("Best LinReg Model: " + str(grid_linreg.best_estimator_))
print("Best Score: " + str(grid_linreg.best_score_))

linreg = grid_linreg.best_estimator_
linreg.fit(X_training, y_training)
lin_pred = linreg.predict(X_valid)

r2_lin =  r2_score(y_valid, lin_pred)

rmse_lin = np.sqrt(mean_squared_error(y_valid, lin_pred))

print("R2 Score"+ str(r2_lin))
print("RMSE Score" + str(rmse_lin))

score_lin = cross_val_score(linreg, X_training, y_training, cv=10, scoring = "r2")

print("Cross validation Score "+ str(np.mean(score_lin)))

#Lasso Model

lasso = Lasso()
parameters_lasso = {"fit_intercept" : [True, False], 
                    "normalize" : [True, False], 
                    "precompute" : [True, False], 
                    "copy_X" : [True, False]}
grid_lasso = GridSearchCV(lasso, parameters_lasso, verbose=1, scoring="r2")
grid_lasso.fit(X_training, y_training)


print("Best Lasso Model: " + str(grid_lasso.best_estimator_))
print("Best Score: " + str(grid_lasso.best_score_))

lasso = grid_lasso.best_estimator_
lasso.fit(X_training, y_training)
lasso_pred = lasso.predict(X_valid)
r2_lasso = r2_score(y_valid, lasso_pred)
rmse_lasso = np.sqrt(mean_squared_error(y_valid, lasso_pred))
print("R^2 Score: " + str(r2_lasso))
print("RMSE Score: " + str(rmse_lasso))

scores_lasso = cross_val_score(lasso, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_lasso)))

#Ridge Model
ridge = Ridge()
parameters_ridge = {"fit_intercept" : [True, False], 
                    "normalize" : [True, False], 
                    "copy_X" : [True, False], 
                    "solver" : ["auto"]}
grid_ridge = GridSearchCV(ridge, parameters_ridge, verbose=1, scoring="r2")
grid_ridge.fit(X_training, y_training)

print("Best Ridge Model: " + str(grid_ridge.best_estimator_))
print("Best Score: " + str(grid_ridge.best_score_))

ridge = grid_ridge.best_estimator_
ridge.fit(X_training, y_training)
ridge_pred = ridge.predict(X_valid)
r2_ridge = r2_score(y_valid, ridge_pred)
rmse_ridge = np.sqrt(mean_squared_error(y_valid, ridge_pred))
print("R^2 Score: " + str(r2_ridge))
print("RMSE Score: " + str(rmse_ridge))

scores_ridge = cross_val_score(ridge, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_ridge)))

#Decision Tree Regressor Model
dtr = DecisionTreeRegressor()
parameters_dtr = {"criterion" : ["mse", "friedman_mse", "mae"], 
                  "splitter" : ["best", "random"], 
                  "min_samples_split" : [2, 3, 5, 10], 
                  "max_features" : ["auto", "log2"]}
grid_dtr = GridSearchCV(dtr, parameters_dtr, verbose=1, scoring="r2")
grid_dtr.fit(X_training, y_training)

print("Best DecisionTreeRegressor Model: " + str(grid_dtr.best_estimator_))
print("Best Score: " + str(grid_dtr.best_score_))

dtr = grid_dtr.best_estimator_
dtr.fit(X_training, y_training)
dtr_pred = dtr.predict(X_valid)
r2_dtr = r2_score(y_valid, dtr_pred)
rmse_dtr = np.sqrt(mean_squared_error(y_valid, dtr_pred))
print("R^2 Score: " + str(r2_dtr))
print("RMSE Score: " + str(rmse_dtr))

scores_dtr = cross_val_score(dtr, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_dtr)))

#Random Forest Regressor
rf = RandomForestRegressor()
paremeters_rf = {"n_estimators" : [5, 10, 15, 20], 
                 "criterion" : ["mse" , "mae"], 
                 "min_samples_split" : [2, 3, 5, 10], 
                 "max_features" : ["auto", "log2"]}
grid_rf = GridSearchCV(rf, paremeters_rf, verbose=1, scoring="r2")
grid_rf.fit(X_training, y_training)

print("Best RandomForestRegressor Model: " + str(grid_rf.best_estimator_))
print("Best Score: " + str(grid_rf.best_score_))

rf = grid_rf.best_estimator_
rf.fit(X_training, y_training)
rf_pred = rf.predict(X_valid)
r2_rf = r2_score(y_valid, rf_pred)
rmse_rf = np.sqrt(mean_squared_error(y_valid, rf_pred))
print("R^2 Score: " + str(r2_rf))
print("RMSE Score: " + str(rmse_rf))

scores_rf = cross_val_score(rf, X_training, y_training, cv=10, scoring="r2")
print("Cross Validation Score: " + str(np.mean(scores_rf)))

#Evaluation Our Models
model_performances = pd.DataFrame({
    "Model" : ["Linear Regression", "Ridge", "Lasso", "Decision Tree Regressor", "Random Forest Regressor"],
    "Best Score" : [grid_linreg.best_score_,  grid_ridge.best_score_, grid_lasso.best_score_, grid_dtr.best_score_, grid_rf.best_score_],
    "R Squared" : [str(r2_lin)[0:5], str(r2_ridge)[0:5], str(r2_lasso)[0:5], str(r2_dtr)[0:5], str(r2_rf)[0:5]],
    "RMSE" : [str(rmse_lin)[0:8], str(rmse_ridge)[0:8], str(rmse_lasso)[0:8], str(rmse_dtr)[0:8], str(rmse_rf)[0:8]]
})
model_performances.round(4)

print("Sorted by Best Score:")
model_performances.sort_values(by="Best Score", ascending=False)
 
print("Sorted by R Squared:")
model_performances.sort_values(by="R Squared", ascending=False)

print("Sorted by RMSE:")
model_performances.sort_values(by="RMSE", ascending=True)

"""
Conclusion: The RMSEs are small because of the log transformation we performed. So even a 0.1 RMSE may be significant in this case.

I decided to choose Random Forest Regressor to use on the test set because I believe it will perform the best based on the statistics printed above. It was a high R^2 value and a lower RMSE.
"""
rf.fit(X_train, y_train)

#Submission
"""
we transformed the Sale Price by taking a log of all the prices? Well, now we need to change that back to the original scale.
 We can do this with numpy's exp function, which will reverse the log. 
 It is the same as raising e to the power of the argument (prediction). (e^pred)
"""        
submission_predictions = np.exp(rf.predict(X_test))

submission = pd.DataFrame({
        "Id": test_data["Id"],
        "SalePrice": submission_predictions
    })

submission.to_csv("prices.csv", index=False)
print(submission.shape)
    














































        
        






