#!/usr/bin/env python
# coding: utf-8

# # Hello, 
# This notebook is how my project goes. 
# this project is called loan prediction using random forest classifier
# 
# ### This project is using an open dataset from Kaggle
# link : https://www.kaggle.com/datasets/kmldas/loan-default-prediction
# 
# This is a synthetic dataset created using actual data from a financial institution. The data has been modified to remove identifiable features and the numbers transformed to ensure they do not link to original source (financial institution). This is intended to be used for academic purposes for beginners who want to practice financial analytics from a simple financial dataset
# 

# In[2]:


# import common package
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler

import warnings 
warnings.filterwarnings('ignore')


# In[4]:


#import data
df = pd.read_csv('Default_Fin.csv', index_col='Index')
df.head()


# # First
# ### we do some simple EDA

# In[5]:


df.info()


# In[7]:


df.shape
# we can see the data has 10.000 records and 4 column


# ## Also we can use some interesting python package to do automated EDA
# Pandas profiling is a Python library that performs an automated Exploratory Data Analysis. It automatically generates a dataset profile report that gives valuable insights. 

# In[8]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df, title="Pandas Profiling Report")
profile.to_widgets()

# we can see the dataset does not have missing value 


# # Create Model
# 
# ### First stage, we can create simple model

# In[9]:


X = df.drop(columns="Defaulted?")
y = df['Defaulted?'] # this is the target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[10]:


## Create Pipeline for Imputer and Transformation

numerical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', RobustScaler())
])

categorical_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])


preprocessor = ColumnTransformer([
    ('numeric', numerical_pipe, ['Bank Balance', 'Annual Salary']),
    ('categoric', categorical_pipe, ["Employed"])
])


# In[11]:


# we can use Logistic Regression
from sklearn.linear_model import LogisticRegression
pipeline = Pipeline([
    ('prep', preprocessor),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])


# In[12]:


from sklearn.model_selection import GridSearchCV

param = {'algo__fit_intercept': [True, False],
         'algo__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
        }

model = GridSearchCV(pipeline, param, cv=3, n_jobs=-1, verbose=1)
model.fit(X_train, y_train)

print(model.best_params_)
print(model.score(X_train, y_train), model.best_score_, model.score(X_test, y_test))


# In[14]:


# Doing evaluation
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix
import matplotlib.pyplot as plt 

y_pred = model.predict(X_test)

print(classification_report(y_test,y_pred))

color = 'black'
matrix = plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


# ## So,
# we can see the Score is really high (97%) but in this case, that score doesn't really matter.
# we must see the f-1 score. F-1 score is really good for predicting 0, why is this happen? this is because imbalance data in target column. So, our model have tendency predict 0 than 1 because it finds 0 more often than 1 when studying the data.
# 
# ### Then what should we do?
# we can do Sampling, for example over sampling with SMOTE 

# In[15]:


# first check imbalance data
ax = sns.countplot(x='Defaulted?',
                 data=df)

for i in ax.containers:
    ax.bar_label(i,)


# In[16]:


# import library for smote
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = smote.fit_resample(X,y)

y_sm.value_counts()


# In[17]:


## train test split again 
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.25, stratify=y_sm, random_state=42)
X_train_sm.shape, X_test_sm.shape, y_train_sm.shape, y_test_sm.shape


# In[19]:


pipeline_sm = Pipeline([
    ('prep', preprocessor),
    ('algo', LogisticRegression(solver='lbfgs', n_jobs=-1, random_state=42))
])

param = {'algo__fit_intercept': [True, False],
         'algo__C': [1.e-03, 1.e-02, 1.e-01, 1.e+00, 1.e+01, 1.e+02, 1.e+03]
        }

model_sm = GridSearchCV(pipeline_sm, param, cv=3, n_jobs=-1, verbose=1)
model_sm.fit(X_train_sm, y_train_sm)

print(model_sm.best_params_)
print(model_sm.score(X_train_sm, y_train_sm), model_sm.best_score_, model_sm.score(X_test_sm, y_test_sm))


# In[22]:


y_pred_sm = model_sm.predict(X_test_sm)

print(classification_report(y_test_sm,y_pred_sm))

color = 'black'
matrix = plot_confusion_matrix(model_sm, X_test_sm, y_test_sm, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


# ## The prediction is getting better
# we can see the f1 score is getting better, what about we using another algorithm
# 
# ## Then what next? maybe we can change the algorithm to the Random Forest Classifier
# Random Forest is strong learner, perhaps we can get a better result.
# Strong learner is such a good things but we must concern that random forest could get overfit, so we must set the parameter to handle overfitting tendency

# In[23]:


from sklearn.ensemble import RandomForestClassifier
param_rf ={'algo__n_estimators': [100, 150, 200],
             'algo__max_depth': [20, 50, 80],
             'algo__max_features': [0.3, 0.6, 0.8],
             'algo__min_samples_leaf': [1, 5, 10]
}

pipeline_sm_rf = Pipeline([
    ('prep', preprocessor),
    ('algo', RandomForestClassifier(n_jobs=-1, random_state=42))
])


model_sm_rf = GridSearchCV(pipeline_sm_rf, param_rf, cv=3, n_jobs=-1, verbose=1)
model_sm_rf.fit(X_train_sm, y_train_sm)


# In[24]:


y_pred_sm = model_sm_rf.predict(X_test_sm)

print(classification_report(y_test_sm,y_pred_sm))

color = 'black'
matrix = plot_confusion_matrix(model_sm_rf, X_test_sm, y_test_sm, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()


# #### As we can see, the model improved based on f1 score. So, we use this algorithm for now

# ## Last Step
# # Save the model

# In[25]:


import pickle 

# save the model to disk
filename = 'model_smote_rf.sav'
pickle.dump(model_sm, open(filename, 'wb'))


# # Build Simple Website with Streamlit
# Next, we build simple website to implement our algorithm with Streamlit Package
# File name : app.py

# In[ ]:




