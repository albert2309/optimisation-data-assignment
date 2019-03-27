# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 03:51:57 2019

@author: alber
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

header = ['age','sex','chest pain','blood pressure', 'serum','fasting','ECG','max heart','angina','oldpeak','slope','major vessels','thal','result']
data = pd.read_csv('heart.dat', names=header, sep=' ', engine='python')
from IPython.display import display
pd.options.display.max_columns = None


#One hot encoding and modify the result so that it can be used for evaluation
X = data.iloc[:,0:13].values
Y = data.iloc[:, 13].values
Y[Y == 1] = 0
Y[Y == 2] = 1
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [2,6,10,12],sparse=False)
encodedData= onehotencoder.fit_transform(X)
encoded_X = pd.DataFrame(encodedData,
             columns=['typical angina chest pain', ' atypical angina chest pain', 'non-anginal chest pain', 
                     'asymptomatic chest pain', 'normal ECG', 'abnormal ECG', 'ventricular hypertrophy ECG',
                      'upsloping slope', 'flat slope', 'downsloping slope',
                      'normal thal', 'fixed defect thal', 'reversable defect thal', 
                      'age', 'sex','blood pressure', 'serum', 'fasting',
                      'max heart', 'angina', 'oldpeak', 'major vessels']
             )

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(encoded_X,Y, test_size = 0.25 , random_state = 123)
# Kera Model

#Alternative 1
# define base model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(10, input_dim=22, kernel_initializer='normal', activation='relu'))
	model.add(Dense(10, input_dim=22, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))
	# Compile model
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model



# evaluate model with standardized dataset
regressor = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=1)

regressor.fit(x_train, y_train, batch_size = 35, epochs = 250, verbose = 1)
y_predict = regressor.predict(x_test)

error = (y_test - y_predict)
RSS = np.sum((error**2))
RSS

mse = np.mean((error)**2)
mse


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Importing the Keras libraries and packages

# for K-fold cross validation we need to wrap keras lassifier into scikitlearn 
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

# one of the arguments for KerasClassifier is a function. The following builds the function
from sklearn.model_selection import GridSearchCV

# def make_classifier(optimizer='adam', nb_epoch = 150):
def make_classifier(optimizer='adam'):
    #Initializing Neural Network
    classifier = Sequential()
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(units= 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))
    # Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

 # here you wrap the keras classifier as a scikit learn function and run grid search
classifier = KerasClassifier(build_fn = make_classifier)
                            
epochs = [150, 250]
batches = [20, 35]
optimizers = ['rmsprop', 'adam']

# Create hyperparameter options
hyperparameters = dict(optimizer=optimizers, nb_epoch=epochs, batch_size=batches)

# params = {
#    'batch_size':[20,35],
#    'nb_epoch':[150,500],
 #   'Optimizer':['adam','rmsprop']
 #   }

grid_search = GridSearchCV(estimator=classifier,
                           param_grid=hyperparameters,
                           cv=10)

grid_search = grid_search.fit(x_train,y_train)

best_param = grid_search.best_params_
best_accuracy = grid_search.best_score_
