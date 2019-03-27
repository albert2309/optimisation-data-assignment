# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 14:36:03 2019

@author: alber
"""

#display(data.head())
#print ("{:<20}{:<8}{:<8}{:<8}{:<8}{:<8}".format('column','min','max','std','mean','median'))
#for x in range(14):  
#    if data.columns.values[x] == 'age' or data.columns.values[x] == 'blood pressure' or \
#    data.columns.values[x] == 'serum' or data.columns.values[x] == 'max heart' or\
#    data.columns.values[x] == 'oldpeak' or data.columns.values[x] == 'major vessels':
#        print ("{:<20}{:<8}{:<8}{:<8}{:<8}{:<8}".format(data.columns.values[x], data.iloc[:, x].min(),
#                   data.iloc[:, x].max(),round(data.iloc[:, x].std(),2),
#                   round(data.iloc[:, x].mean(),2),round(data.iloc[:, x].median(),2)))
##Printing the 
#for x in range(14):
#    if data.columns.values[x] != 'age' and data.columns.values[x] != 'blood pressure' and \
#    data.columns.values[x] != 'serum' and data.columns.values[x] != 'max heart' and\
#    data.columns.values[x] != 'oldpeak' and data.columns.values[x] != 'major vessels':
#        display(data.groupby(data.columns.values[x]).size().reset_index(name='Count').rename(columns={'Col1':'Col_value'}))
#data.boxplot(by='result',column=['age'])
#data.boxplot(by='result',column=['blood pressure'])
#import matplotlib.pyplot as plt

#hasDisease = data['result'] == 1
#filteredScatterPlot= sns.FacetGrid(data[hasDisease], hue="sex", height=10) \
#   .map(plt.scatter, "age", "chest pain") \
#   .add_legend()
#filteredScatterPlot._legend.set_title('Gender')
#new_labels = ['Female', 'Male']
#for t, l in zip(filteredScatterPlot._legend.texts, new_labels): t.set_text(l)

#filteredScatterPlot= sns.FacetGrid(data, hue="result", size=10) \
#   .map(plt.scatter, "blood pressure", "serum") \
#   .add_legend()
#new_labels = ['Absent', 'Present']
#for t, l in zip(filteredScatterPlot._legend.texts, new_labels): t.set_text(l)
#heatmaptest = pd.pivot_table(data, values = 'result',index=['age'], columns = 'age')
#sns.heatmap(heatmaptest, annot=True, linewidths=.5)
#data.iloc[:,0:4].values
#

#pair plot
#sns.pairplot(data, vars= ['serum','max heart'], size=3,hue='result')
#No correlation between them
#data.plot(kind='scatter',x='age', y='blood pressure')


#import seaborn as sns; sns.set(style="ticks", color_codes=True)        
#import matplotlib.pyplot as plt
#g  = sns.FacetGrid(data, row="slope",hue='result')
#g= g.map(plt.hist, "blood pressure").add_legend()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


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


alpha_range = [0.1, 0.5, 0.8, 1.0, 1.3, 1.4, 1.8, 2, 3, 10, 20, 30, 40, 50, 100, 200]
#k_fold = KFold(3)

#param_grid = {'alpha': alpha_range}
## param_grid = np.array(param_grid)
## print (param_grid)
#grid = LassoCV(alphas=alpha_range, cv=5, random_state=123)
#for k, (train, test) in enumerate(k_fold.split(encoded_X, Y)):
#    grid.fit(encoded_X[train], Y[train])
#    print(k)
#    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
#          format(k, grid.alpha_, grid.score(encoded_X[test], encoded_X[test])))

#grid.fit(encoded_X,Y)
#grid.predict(encoded_X)
##print(grid.best_score_)
##print(grid.best_params_)
#grid.grid_scores_
#grid.grid_scores_[0].parameters
#grid.grid_scores_[0].cv_validation_scores
#grid_mean_scores = [result.mean_validation_score for result in grid.grid_scores_]
#
#grid.grid_scores_[0].mean_validation_score
#plt.plot(alpha_range, grid_mean_scores) 
    
    
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
lasso = Lasso(random_state=123)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(encoded_X, Y)
scores = clf.cv_results_['mean_test_score']
scores_std = clf.cv_results_['std_test_score']
plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores
std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, 'b--')
plt.semilogx(alphas, scores - std_error, 'b--')

# alpha=0.2 controls the translucency of the fill color
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel('CV score +/- std error')
plt.xlabel('alpha')
plt.axhline(np.max(scores), linestyle='--', color='.5')
plt.xlim([alphas[0], alphas[-1]])


lasso_cv = LassoCV(alphas=alphas, cv=5, random_state=123)
k_fold = KFold(3)
#values is used to convert datafraom to numpy array
for k, (train, test) in enumerate(k_fold.split(encoded_X, Y)):
    lasso_cv.fit(encoded_X.values[train], Y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(encoded_X.values[test], Y[test])))