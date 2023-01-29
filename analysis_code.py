# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 20:56:03 2023

@author: vimto
"""

#this is a short dataanalysis of the churn rate from a bank.
#The dataset is downloaded from Kaggle, you can use the below link to download the dataset.
#https://www.kaggle.com/datasets/shubh0799/churn-modelling?resource=download
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import make_pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import f1_score
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import classification_report
from sklearn.ensemble import VotingClassifier

from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
#change directory to where the data is located
os.chdir("C:/Users/vimto/Desktop/churndata")

df = pd.read_csv("Churn_Modelling.csv")

#initial look at the data

print(df.columns)
#Index(['RowNumber', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
  #     'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
   #    'IsActiveMember', 'EstimatedSalary', 'Exited'],
    #  dtype='object')

#customerId and RowNumber does not carry interesting information

#exited denotes if a customer will leave, so it is the variable we want to predict


#Create dummy variables such that I can insert Categorical variables into ML models


df_temp = pd.get_dummies(df,columns = ["Geography","Gender"])

#gender_male and Geography_Germany contain redundant information since you can tell them from the other variables
df_temp = df_temp.drop(columns=["RowNumber","CustomerId","Gender_Male","Geography_Germany"])


#there are around 3000 different surnames
#a lot of people share surnames, which could be useful if families quit/stay together,
#a source of error here is that some surnames are very common

#I check the importance of surnames in the following histogram
ax,fig = plt.subplots()
fig.set_title("Churn rate for different surnames")
fig.set_xlabel("Churn rate")
#I want to avoid single individuals
indx = df_temp.groupby("Surname").count()['CreditScore']>1
fig.hist(df_temp.groupby("Surname").agg("mean")["Exited"][indx],bins=50)


#the largest bins have a mean churn rate of 0 and most families have a low churn rate, but this is a small amount of observations
#it is also worth noting that the number of people that leaves is 1/5 of the dataset
#this could be useful for a nearest neighbor classifier which could be used in an ensemble

#distribution accross countries, France is about half the observations, Spain and Germany have similar shares
print("France percentage: "+str(df_temp["Geography_France"].sum()/len(df_temp)))

print("Spain percentage: "+str(df_temp["Geography_Spain"].sum()/len(df_temp)))

print("Germany percentage: " + str(1 -  df_temp["Geography_France"].sum()/len(df_temp) - df_temp["Geography_Spain"].sum()/len(df_temp) ))

#heatmap
H = df_temp.corr()
plt.imshow(H)
plt.colorbar()
plt.tick_params(axis="x", rotation=75)
plt.xticks(range(len(H)), H.columns)
plt.yticks(range(len(H)), H.columns)
plt.show()
#the most interesting relation is a negative correlation between balance and NumOfProducts
#the largest correlation values for the exited variable (which we want to predict) are 0.28 for age, -0.23 balance, -0.16 IsActive, -0.1 for France and 0.1 for Gender_Female
#not large correlations but maybe they can be combined to get useful predictions 

#balance and NumOfProducts also have a relatively large correlation
ax,fig = plt.subplots()
fig.set_xlabel("NumOfProducts")
fig.set_ylabel("Balance")
fig.plot(df_temp.groupby("NumOfProducts").agg("mean")["Balance"])
plt.show()
#people with 2 products seem to have a much lower balance
#people with 3 and 4 products take up a much smaller part, smaller than 5% of observations
#perhabs NumOfProducts should be treated as a categorical variable instead of a hierachal
print(df_temp.groupby("NumOfProducts").count()['Surname']/len(df_temp))

#count/distribution check of the other variables

print(df_temp.groupby("IsActiveMember").count()['Surname']/len(df_temp))

print(df_temp.groupby("HasCrCard").count()['Surname']/len(df_temp))



#Time to try some models

x = df_temp.drop(columns=["Surname","Exited"])
x = pd.get_dummies(x,columns = ["NumOfProducts"])
y = df_temp["Exited"]

#this is a function for plotting mean scores as a function of parameter values, an interval is also plotted to denote the mean +- the standard deviation
#model is an object storing the results of a gridsearch with control validation, param_str is the string for the parameter and score_str is a string denoting the score
def plot_cv_1d(model,param_str,score_str):
    
    param_grid = model.param_grid
    
    ax,fig = plt.subplots()
    plt.plot(param_grid[param_str],model.cv_results_["mean_test_"+score_str])
    plt.plot(param_grid[param_str],model.cv_results_["mean_test_"+score_str]-model.cv_results_["std_test_"+score_str])
    plt.plot(param_grid[param_str],model.cv_results_["mean_test_"+score_str]+model.cv_results_["std_test_"+score_str])
    fig.set_xlabel(param_str)
    fig.set_ylabel(score_str)
    plt.show()


#I try to fit two classification trees, one where observations are weighted in the tree to account for the dataset imbalance
w_stay = 0.5/7963
w_exit = 0.5/2037
weights = np.zeros((len(y)))
weights[y==1] = w_exit
weights[y==0] = w_stay

param_grid = {"ccp_alpha": np.exp(np.arange(-10,-5,0.1))}
model = GridSearchCV(estimator = tree.DecisionTreeClassifier(),param_grid = param_grid,cv=10,refit="f1",scoring = ["f1","accuracy","recall"])
model.fit(x,y,sample_weight=weights)
model.best_estimator_

plot_cv_1d(model,"ccp_alpha","f1")

plot_cv_1d(model,"ccp_alpha","accuracy")

plot_cv_1d(model,"ccp_alpha","recall")

plt.figure(figsize=(18,18),dpi=400)
tree.plot_tree(model.best_estimator_,feature_names = list(x.columns),class_names=["Stayed","Exited"],filled=True)

param_grid = {"ccp_alpha": np.exp(np.arange(-15,-5,0.1))}
model = GridSearchCV(estimator = tree.DecisionTreeClassifier(),param_grid = param_grid,cv=10,refit="f1",scoring = ["f1","accuracy","recall"])
model.fit(x,y)
model.best_estimator_

plot_cv_1d(model,"ccp_alpha","f1")

plot_cv_1d(model,"ccp_alpha","accuracy")

plot_cv_1d(model,"ccp_alpha","recall")

plt.figure(figsize=(18,18),dpi=400)
tree.plot_tree(model.best_estimator_,feature_names = list(x.columns),class_names=["Stayed","Exited"],filled=True)
#weighting samples gives us a significantly higher recall at a loss of accuracy, at a glance then the f1 score is just below 0.58

#age, isactive and the number of products seem to be the most important features

#looks like there might be a second order effect or higher

x = df_temp.drop(columns=["Surname","Exited"])
x = pd.get_dummies(x,columns = ["NumOfProducts"])

poly = PolynomialFeatures(2)


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=42)

model_lda = make_pipeline(poly,StandardScaler(),LDA())
model_lda.fit(X_train,y_train)
model_lda.score(X_test,y_test)

y_pred = model_lda.predict(X_test)
f1_score(y_test,y_pred)
#results are improved by adding the age x active interaction
#model could probably be improved by testing and adding more interactions

model_log= make_pipeline(poly,StandardScaler(),logreg(max_iter = 1000))
model_log.fit(X_train,y_train)
model_log.score(X_test,y_test)
y_pred = model_log.predict(X_test)
f1_score(y_test,y_pred)

#these models don't seem to offer better results than the decision trees

#here is a custom estimator that is gonna use the surname feature

class MyEstimator(BaseEstimator,ClassifierMixin):
    def __init__(self,param1 = None,param2=None):
        self.param1 = param1
        self.param2 = param2
        
    def fit(self,x,y = None):
        self.param1 = pd.DataFrame(x["Surname"].copy())
        self.param1["Exited"] = y
    
        self.param1 = self.param1.groupby("Surname").agg("mean")
    
        self.param2 = y.mean()
        self.X_ = x
        self.y_ = y
        
        return(self)
    
    def predict(self,x,y=None):
    
        check_is_fitted(self)
    
    
        out = pd.merge(x["Surname"],self.param1,how="left",right_index=True,left_on="Surname")
        out = out.fillna(value = self.param2)
    
        return np.array(out["Exited"] > 0.5,dtype="int64")
    
    def predict_proba(self, x, y=None):
        
        check_is_fitted(self)
        
        out = pd.merge(x["Surname"],self.param1,how="left",right_index=True,left_on="Surname")
        out = out.fillna(value = self.param2)
    
        return np.column_stack([1-out["Exited"],out["Exited"]])
    
    def score(self,y,y_pred):
        
        return(y == y_pred).sum()/len(y)


class dropSurname(BaseEstimator):
    def __init__(self,param1 = None):
        self.param1 = param1
        
    def fit(self,x,y=None):
        return(self)
    def transform(self,x):
        
        return(x.drop(columns = ["Surname"]))





pipe_fam = make_pipeline(MyEstimator())

pipe_forest = make_pipeline(dropSurname(),RandomForestClassifier())


v=VotingClassifier(estimators=[('family_model',pipe_fam),("RandomForest",pipe_forest)],voting  = "soft")

#I am now using surname
x = df_temp.drop(columns=["Exited"])
x = pd.get_dummies(x,columns = ["NumOfProducts"])
y = df_temp["Exited"]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


param_dict = {'RandomForest__randomforestclassifier__n_estimators': [50,100,200], 'RandomForest__randomforestclassifier__ccp_alpha': [0.0005,0.001,0.002],"weights": [(0,1),(0.25,0.75),(0.5,0.5),(0.75,0.25),(1,0)]}


model = GridSearchCV(estimator =v,param_grid = param_dict,cv=10,refit="f1",scoring = ["f1","accuracy","recall"])
model.fit(X_train,y_train)
print(classification_report(y_test,model.best_estimator_.predict(X_test)))


#random forest without using the Surname features
x = df_temp.drop(columns=["Surname","Exited"])
x = pd.get_dummies(x,columns = ["NumOfProducts"])
y = df_temp["Exited"]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)


#no improvements using the Surname data
rf_model = RandomForestClassifier()
param_dict = {'n_estimators': [50,100,200], 'ccp_alpha': [0.0005,0.001,0.002],"max_features": [1,5,10]}

model = GridSearchCV(estimator =rf_model,param_grid = param_dict,cv=10,refit="f1",scoring = ["f1","accuracy","recall"])
model.fit(X_train,y_train)
print(classification_report(y_test,model.best_estimator_.predict(X_test)))

#random forest with an oversampling strategy

X_res,y_res = SMOTE().fit_resample(X_train,y_train)
rf_model = RandomForestClassifier()
param_dict = {'n_estimators': [50,100,200], 'ccp_alpha': [0.0005,0.001,0.002],"max_features": [1,5,10]}

model = GridSearchCV(estimator =rf_model,param_grid = param_dict,cv=10,refit="f1",scoring = ["f1","accuracy","recall"])
model.fit(X_res,y_res)
print(classification_report(y_test,model.best_estimator_.predict(X_test)))


rf_model = RandomForestClassifier()
param_dict = {'n_estimators': [100], 'max_depth': [1,2,3,6,10,14],"max_features": [10],'min_samples_leaf': [1,2,5,10,20]}

model = GridSearchCV(estimator =rf_model,param_grid = param_dict,cv=10,refit="recall",scoring = ["f1","accuracy","recall"])
w_stay = 0.5/7963
w_exit = 0.5/2037
weights = np.zeros((len(y_train)))
weights[y_train==1] = w_exit
weights[y_train==0] = w_stay
model.fit(X_train,y_train,sample_weight=weights)
print(classification_report(y_test,model.best_estimator_.predict(X_test)))

#the best model I found was a randomforest classifier with weighted samples

#A well performing Kaggle model gets a recall around 78% and accuracy around 80%, the below code is an attempt at verifying their results using my framework
#I couldn't get the same results, but since the randomforest classifier with weighted samples achieves a recall and accuracy in that range, then I am confident it is possible
#https://www.kaggle.com/code/lmarcer/churn-smote-78-recall-80-accuracy
#from xgboost import XGBClassifier
#from sklearn.preprocessing import StandardScaler
#from imblearn.over_sampling import SMOTE
#x = df_temp.drop(columns=["Surname","Exited"])
#x = pd.get_dummies(x,columns = ["NumOfProducts"])
#y = df_temp["Exited"]
#X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
#X_res,y_res = SMOTE().fit_resample(X_train,y_train)

#rf_model=XGBClassifier()


#param_dict = {"n_estimators": [1,10,100,200],"max_depth": [1,2,3,6,7,8,9,10],"eta": [0.0001,0.001,0.01,0.1,1],"colsample_bylevel": [0.1,0.5,1]}
#model = GridSearchCV(estimator =rf_model,param_grid = param_dict,cv=5,refit="f1",scoring = ["f1","accuracy","recall"])
#model.fit(X_res,y_res)

#pred=model.best_estimator_.predict(X_test)
#print(classification_report(y_test,pred))



X_res,y_res = SMOTE().fit_resample(X_train,y_train)
params2 = { "objective":"binary:logistic",'colsample_bytree':0.5,"learning_rate": 0.025,
                'max_depth': 4,"gamma": 0.3, "subsample": 0.66,  'n_estimators':1000, 'min_child_weight':5 }
classifier = XGBClassifier(**params2)

classifier=classifier.fit(X_res,y_res)
pred=classifier.predict(X_test)
print(classification_report(y_test,pred))