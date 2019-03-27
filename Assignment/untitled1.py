# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 13:06:41 2019

@author: alber
"""

## To do
""" 
1. Select you preferred datset. Ex: prostate datset, diabetes or Boston or 
any other dataset with around 10-15 input variables.
 2. Perform EDA
 3. Perform Preprocessing
EDA tasklist:
 1. mean
 2. median
 3. Box plot
 4. Correlation (use scatter plot and heat map)
 """
import numpy as np
import pandas as pd
import seaborn as sns

"""
Attributes types 
----------------- 
Real: 1,4,5,8,10,12 
Ordered:11, 
Binary: 2,6,9 
Nominal:7,3,13 
      -- 1. age       
      -- 2. sex       
      -- 3. chest pain type  (4 values)       
      -- 4. resting blood pressure  
      -- 5. serum cholestoral in mg/dl      
      -- 6. fasting blood sugar > 120 mg/dl       
      -- 7. resting electrocardiographic results  (values 0,1,2) 
      -- 8. maximum heart rate achieved  
      -- 9. exercise induced angina    
      -- 10. oldpeak = ST depression induced by exercise relative to rest   
      -- 11. the slope of the peak exercise ST segment     
      -- 12. number of major vessels (0-3) colored by flourosopy        
      -- 13.  thal
      -- 14. result
    Attributes:
        2. sex 
        -- 1 = male 
        -- 0 = female
        3. chest pain type
        -- 1 = typical angina 
        -- 2 = atypical angina 
        -- 3 = non-anginal pain 
        -- 4 = asymptomatic 
        6. fasting blood sugar 
        -- 1 = true
        -- 0 = false
        7. resting electrocardiographic results
        -- 0 = normal 
        -- 1 = having ST-T wave abnormality (T wave inversions
        and/or ST elevation or depression of > 0.05 mV) 
        -- 2: showing probable or definite left 
        ventricular hypertrophy by Estes' criteria 
        9.  exercise induced angina 
        -- 1 = yes
        -- 0 = no
        11. slope
        -- 1 = upsloping 
        -- 2 = flat 
        -- 3 = downsloping 
        13. thal
        -- 3 = normal
        -- 6 = fixed defect
        -- 7 = reversable defect
        14. result
        -- 1 = Absence of heart disease
        -- 2 = Presence of heart disease

major vessels (0-3) colored by flourosopy => larger number mean higher chance of getting diseases

#JUST SHOW CORRELATION BETWEEN A CLASS AN ATTRIBUTE OR AN ATTRIBURTE AND ANTTRIBUTE
#Try regularization (effect of an aattribute on the output), neural network, SVM, and linear regression. No naive bayes.       
#try to understand grid search and cross-validation (must be bring in the experiment)
#Try all models in your own dataset and write the report.
#Literature Review 1 page is enough
try: tf.info(())
"""

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

#4. Fitting Logistic regression to the Training set
from sklearn.linear_model import LogisticRegression, Lasso
# c= regularizaiton
classifier = LogisticRegression(C = 0.2, random_state = 123)
classifier.fit(x_train, y_train)

#5. Predict the result of the madel
y_pred = classifier.predict(x_test)
print("First ten prediction result")
print(classifier.predict_proba(x_test)[:10])
print("Accuracy score: " + str(classifier.score(x_test,y_test)))
print("number of features used:" + str(np.sum(classifier.coef_!=0)))
THRESHOLD = 0.5
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
print("Confusion matrix for those who have diseases")
preds = np.where(classifier.predict_proba(x_test)[:,1] > THRESHOLD, 1, 0)
display(pd.DataFrame(data=[accuracy_score(y_test, preds), recall_score(y_test, preds),
                   precision_score(y_test, preds), roc_auc_score(y_test, preds)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"]))


cm = confusion_matrix(y_test, preds).ravel()
print("True Negative : " + str(cm[0]))
print("False Positive : " + str(cm[1]))
print("False Negative : " + str(cm[2]))
print("True Positive : " + str(cm[3]))


predictors = x_train.columns
coef = pd.Series(classifier.coef_[0],predictors).sort_values()
coef.plot(kind='bar',title='Modal Coefficients')
display(coef)

## To do: Plot ROC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#Step 2. Compute ROC curve and ROC area for each class
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)

##Step 3. Plot the figure
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label ='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC - ROC Curve')
plt.legend(loc="lower right")
plt.show()


#Part 2: Lasso
lassoClassifier = Lasso(alpha=0.004893900918477494, random_state = 123)
lassoClassifier.fit(x_train, y_train)

y_pred = lassoClassifier.predict(x_test)
print("Accuracy score for lasso classifier: " + str(lassoClassifier.score(x_test,y_test)))
print("number of features used:" + str(np.sum(lassoClassifier.coef_!=0)))

y_pred_categorised = np.where(lassoClassifier.predict(x_test) > THRESHOLD, 1, 0)
display(pd.DataFrame(data=[accuracy_score(y_test, y_pred_categorised), recall_score(y_test, y_pred_categorised),
                   precision_score(y_test, y_pred_categorised), roc_auc_score(y_test, y_pred_categorised)], 
             index=["accuracy", "recall", "precision", "roc_auc_score"]))

cm = confusion_matrix(y_test, y_pred_categorised).ravel()
print("True Negative : " + str(cm[0]))
print("False Positive : " + str(cm[1]))
print("False Negative : " + str(cm[2]))
print("True Positive : " + str(cm[3]))

## To do: Plot ROC
#Step 2. Compute ROC curve and ROC area for each class
fpr, tpr, threshold = roc_curve(y_test, y_pred_categorised)
roc_auc = auc(fpr, tpr)

##Step 3. Plot the figure
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label ='ROC Curve (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC - ROC Curve')
plt.legend(loc="lower right")
plt.show()

predictors = x_train.columns
coef = pd.Series(lassoClassifier.coef_,predictors).sort_values()
coef.plot(kind='bar',title='Modal Coefficients')