#Import Libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor 


#Read the data.
data = pd.read_csv('general_data.csv')
emp_survey = pd.read_csv('employee_survey_data.csv')
man_survey = pd.read_csv('manager_survey_data.csv')

#Merge datasets
df = pd.merge(pd.merge(data,emp_survey,on='EmployeeID'),man_survey,on='EmployeeID')
#Mapping numerical values to attrition.
df['Attrition'] = df['Attrition'].map({'No':0, 'Yes':1})

#Function to preprocess.
# def pre_process(df):
#     df.dropna(inplace=True)
#     df = pd.get_dummies(df)
#     return df
    
#Drop NA
df.dropna(inplace=True)
#Dropping Employee Count and StandardHours features (sd=0/have just one value in column).
df.drop(['EmployeeCount','YearsAtCompany','TotalWorkingYears','StandardHours','EmployeeID','Over18'],axis=1 , inplace=True)
#Get Dummies
df = pd.get_dummies(df)
#Predictors
X = df.drop('MonthlyIncome',axis=1)
#Response Feature
y = df['MonthlyIncome']

# #Split the dataset into 80-20.
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)

#Applying Scaler.
scaler = preprocessing.Normalizer()
X = scaler.fit_transform(X)
# X_train = scaler.fit_transform(X_train) #notice how the target feature (y) is untouched.
# X_test = scaler.fit_transform(X_test)

# Fit regression model using best params.
params = {'n_estimators': 500, 'max_depth': 8, 'min_samples_split': 20,
          'learning_rate': 0.1, 'loss': 'ls'}
clf = GradientBoostingRegressor(**params)
clf.fit(X, y)

#Saving model to disk
pickle.dump(clf, open('model.pkl','wb'))

#Saving Scaler
pickle.dump(scaler, open('scaler.pkl','wb'))

# Loading model and scaler to compare the results
model = pickle.load(open('model.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

#Sample Prediction (Values scaled already).
print(model.predict([[0.89033932, 0.        , 0.02406322, 0.0962529 , 0.04812645,
       0.16844257, 0.33688515, 0.        , 0.0962529 , 0.        ,
       0.0962529 , 0.0962529 , 0.0962529 , 0.04812645, 0.07218967,
       0.07218967, 0.        , 0.        , 0.02406322, 0.        ,
       0.        , 0.02406322, 0.        , 0.02406322, 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.02406322,
       0.02406322, 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.02406322,
       0.        , 0.        ]]))


