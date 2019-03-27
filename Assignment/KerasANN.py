#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:54:58 2019

@author: rajemandava
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 27 18:15:40 2018

@author: rajemandava
"""

import numpy as np

import pandas as pd

from pandas import Series, DataFrame


from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

# Data preparation
mart = pd.read_csv('mart.csv')
# imputing missing values

mart['Item_Weight'].fillna((mart['Item_Weight'].mean()), inplace=True)
mart['Item_Visibility'] = mart['Item_Visibility'].replace(0,np.mean(mart['Item_Visibility']))
mart['Outlet_Establishment_Year'] = 2013 - mart['Outlet_Establishment_Year']

mart['Outlet_Size'].fillna('Small',inplace=True)
mart = mart.iloc [ : , 1: 12]
# creating dummy variables to convert categorical into numeric values
mylist = list(mart.select_dtypes(include=['object']).columns)
dummies = pd.get_dummies(mart[mylist], prefix= mylist)
mart.drop(mylist, axis=1, inplace = True)
X = pd.concat([mart,dummies], axis =1 )
y = X.iloc [ :, 4 ]
# drop the response variable
X = X.drop('Item_Outlet_Sales',1)

# creating training and testing data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Kera Model

#Alternative 1
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=45, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, input_dim=45, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



# evaluate model with standardized dataset
regressor = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)

regressor.fit(X_train, y_train, batch_size = 5, epochs = 100, verbose = 1)
y_predict = regressor.predict(X_test)

error = (y_test - y_predict)
RSS = np.sum((error**2))
RSS

mse = np.mean((error)**2)
mse
