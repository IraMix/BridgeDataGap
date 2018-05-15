#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 19:13:35 2018

@author: iramixfiles test
"""

import os
import pandas as pd
import numpy as np
import matplotlib as plt

path='/Users/iramixfiles/Documents/BDG/PythonGitHub/BridgeDataGap'
os.chdir(path)

plt.pyplot.plot(np.arange(5))

df = pd.read_csv("train.csv") #Reading the dataset in a dataframe using Pandas

df.head()
df.dtypes

df.describe()

df['Property_Area'].value_counts()

#Distribution analysis

#Now that we are familiar with basic data characteristics, let us study distribution of various variables. Let us start with numeric variables – namely ApplicantIncome and LoanAmount

#Lets start by plotting the histogram of ApplicantIncome using the following commands:

df['ApplicantIncome'].hist(bins=50)

#Here we observe that there are few extreme values. This is also the reason why 50 bins are required to depict the distribution clearly.

#Next, we look at box plots to understand the distributions. Box plot for fare can be plotted by:

df.boxplot(column='ApplicantIncome') # Cjearly outliers

#This confirms the presence of a lot of outliers/extreme values. This can be attributed to the income disparity in the society. Part of this can be driven by the fact that we are looking at people with different education levels. Let us segregate them by Education:

df.boxplot(column='ApplicantIncome', by = 'Education')

#We can see that there is no substantial different between the mean income of graduate and non-graduates. But there are a higher number of graduates with very high incomes, which are appearing to be the outliers.

#Now, Let’s look at the histogram and boxplot of LoanAmount using the following command:

df['LoanAmount'].hist(bins=50)

df.boxplot(column='LoanAmount')

#Again, there are some extreme values. Clearly, both ApplicantIncome and LoanAmount require some amount of data munging. LoanAmount has missing and well as extreme values values, while ApplicantIncome has a few extreme values, which demand deeper understanding. We will take this up in coming sections.

 
#Categorical variable analysis

#Now that we understand distributions for ApplicantIncome and LoanIncome, let us understand categorical variables in more details. We will use Excel style pivot table and cross-tabulation. For instance, let us look at the chances of getting a loan based on credit history. This can be achieved in MS Excel using a pivot table as:

#Note: here loan status has been coded as 1 for Yes and 0 for No. So the mean represents the probability of getting loan.

#Now we will look at the steps required to generate a similar insight using Python. Please refer to this article for getting a hang of the different data manipulation techniques in Pandas.

temp1 = df['Credit_History'].value_counts(ascending=True)
temp2 = df.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())

print ('Frequency Table for Credit History:' )
print (temp1)

print ('\nProbility of getting loan for each Credit History class:' )
print (temp2)

#Now we can observe that we get a similar pivot_table like the MS Excel one. This can be plotted as a bar chart using the “matplotlib” library with following code:

#This shows that the chances of getting a loan are eight-fold if the applicant has a valid credit history. You can plot similar graphs by Married, Self-Employed, Property_Area, etc.

#Alternately, these two plots can also be visualized by combining them in a stacked chart::

temp3 = pd.crosstab(df['Credit_History'], df['Loan_Status'])
temp3.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)

#You can also add gender into the mix (similar to the pivot table in Excel):

#If you have not realized already, we have just created two basic classification algorithms here, one based on credit history, while other on 2 categorical variables (including gender). You can quickly code this to create your first submission on AV Datahacks.

#We just saw how we can do exploratory analysis in Python using Pandas. I hope your love for pandas (the animal) would have increased by now – given the amount of help, the library can provide you in analyzing datasets.

#Next let’s explore ApplicantIncome and LoanStatus variables further, perform data munging and create a dataset for applying various modeling techniques. I would strongly urge that you take another dataset and problem and go through an independent example before reading further.

 

#4. Data Munging in Python : Using Pandas

#For those, who have been following, here are your must wear shoes to start running.

#Data munging – recap of the need

#While our exploration of the data, we found a few problems in the data set, which needs to be solved before the data is ready for a good model. This exercise is typically referred as “Data Munging”. Here are the problems, we are already aware of:

#There are missing values in some variables. We should estimate those values wisely depending on the amount of missing values and the expected importance of variables.

#While looking at the distributions, we saw that ApplicantIncome and LoanAmount seemed to contain extreme values at either end. Though they might make intuitive sense, but should be treated appropriately.


#In addition to these problems with numerical fields, we should also look at the non-numerical fields i.e. Gender, Property_Area, Married, Education and Dependents to see, if they contain any useful information.


 

#Check missing values in the dataset

#Let us look at missing values in all the variables because most of the models don’t work with missing data and even if they do, imputing them helps more often than not. So, let us check the number of nulls / NaNs in the dataset

df.apply(lambda x: sum(x.isnull()),axis=0) 
 
#This command should tell us the number of missing values in each column as isnull() returns 1, if the value is null.

#Though the missing values are not very high in number, but many variables have them and each one of these should be estimated and added in the data. Get a detailed view on different imputation techniques through this article.

#Note: Remember that missing values may not always be NaNs. For instance, if the Loan_Amount_Term is 0, does it makes sense or would you consider that missing? I suppose your answer is missing and you’re right. So we should check for values which are unpractical.

 

#How to fill missing values in LoanAmount?

## First way to fill missing Values

#There are numerous ways to fill the missing values of loan amount – the simplest being replacement by mean, which can be done by following code:

df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)

#The other extreme could be to build a supervised learning model to predict loan amount on the basis of other variables and then use age along with other variables to predict survival.

#Since, the purpose now is to bring out the steps in data munging, I’ll rather take an approach, which lies some where in between these 2 extremes. A key hypothesis is that the whether a person is educated or self-employed can combine to give a good estimate of loan amount.

#First, let’s look at the boxplot to see if a trend exists:

#Thus we see some variations in the median of loan amount for each group and this can be used to impute the values. But first, we have to ensure that each of Self_Employed and Education variables should not have a missing values.

#As we say earlier, Self_Employed has some missing values. Let’s look at the frequency table:

#Since ~86% values are “No”, it is safe to impute the missing values as “No” as there is a high probability of success. This can be done using the following code:


df['Self_Employed'].fillna('No',inplace=True)


#Second Way of replacing values
#Now, we will create a Pivot table, which provides us median values for all the groups of unique values of Self_Employed and Education features. Next, we define a function, which returns the values of these cells and apply it to fill the missing values of loan amount:

table = df.pivot_table(values='LoanAmount', index='Self_Employed' ,columns='Education', aggfunc=np.median)

table

# Define function to return value of this pivot_table
def fage(x):
 return table.loc[x['Self_Employed'],x['Education']]
 
# Replace missing values
df['LoanAmount'].fillna(df[df['LoanAmount'].isnull()].apply(fage, axis=1), inplace=True)


#This should provide you a good way to impute missing values of loan amount.

 

#How to treat for extreme values in distribution of LoanAmount and ApplicantIncome?

#Let’s analyze LoanAmount first. Since the extreme values are practically possible, i.e. some people might apply for high value loans due to specific needs. So instead of treating them as outliers, let’s try a log transformation to nullify their effect:

df['LoanAmount_log'] = np.log(df['LoanAmount'])
df['LoanAmount_log'].hist(bins=20)

#Looking at the histogram again:

df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log(df['TotalIncome'])
df['LoanAmount_log'].hist(bins=20) 

#Now we see that the distribution is much better than before. I will leave it upto you to impute the missing values for Gender, Married, Dependents, Loan_Amount_Term, Credit_History. Also, I encourage you to think about possible additional information which can be derived from the data. For example, creating a column for LoanAmount/TotalIncome might make sense as it gives an idea of how well the applicant is suited to pay back his loan.

#Next, we will look at making predictive models.

 

#5. Building a Predictive Model in Python

#After, we have made the data useful for modeling, let’s now look at the python code to create a predictive model on our data set. Skicit-Learn (sklearn) is the most commonly used library in Python for this purpose and we will follow the trail. I encourage you to get a refresher on sklearn through this article.

#Since, sklearn requires all inputs to be numeric, we should convert all our categorical variables into numeric by encoding the categories. This can be done using the following code:

from sklearn.preprocessing import LabelEncoder
var_mod = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
le = LabelEncoder()
for i in var_mod:
    df[i] = le.fit_transform(df[i])
df.dtypes 

#Next, we will import the required modules. Then we will define a generic classification function, which takes a model as input and determines the Accuracy and Cross-Validation scores. Since this is an introductory article, I will not go into the details of coding. Please refer to this article for getting details of the algorithms with R and Python codes. Also, it’ll be good to get a refresher on cross-validation through this article, as it is a very important measure of power performance.

 
#Import models from scikit learn module:
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics

#Generic function for making a classification model and accessing performance:
def classification_model(model, data, predictors, outcome):
  
    #Fit the model:
  model.fit(data[predictors],data[outcome])
  
  #Make predictions on training set:
  predictions = model.predict(data[predictors])
  
  #Print accuracy
  accuracy = metrics.accuracy_score(predictions,data[outcome])
  print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

  #Perform k-fold cross-validation with 5 folds
  kf = KFold(data.shape[0], n_folds=5)
  error = []
  for train, test in kf:
    # Filter training data
    train_predictors = (data[predictors].iloc[train,:])
    
    # The target we're using to train the algorithm.
    train_target = data[outcome].iloc[train]
    
    # Training the algorithm using the predictors and target.
    model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
    error.append(model.score(data[predictors].iloc[test,:], data[outcome].iloc[test]))
 
  print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))

  #Fit the model again so that it can be refered outside the function:
  model.fit(data[predictors],data[outcome]) 
  
#Logistic Regression

#Let’s make our first Logistic Regression model. One way would be to take all the variables into the model but this might result in overfitting (don’t worry if you’re unaware of this terminology yet). In simple words, taking all variables might result in the model understanding complex relations specific to the data and will not generalize well. Read more about Logistic Regression.

#We can easily make some intuitive hypothesis to set the ball rolling. The chances of getting a loan will be higher for:

#Applicants having a credit history (remember we observed this in exploration?)
#Applicants with higher applicant and co-applicant incomes
#Applicants with higher education level
#Properties in urban areas with high growth perspectives
#So let’s make our first model with ‘Credit_History’.

outcome_var = 'Loan_Status'
model = LogisticRegression()
predictor_var = ['Credit_History']
classification_model(model, df,predictor_var,outcome_var)
#Accuracy : 80.945% Cross-Validation Score : 80.946%

#We can try different combination of variables:
predictor_var = ['Credit_History','Education','Married','Self_Employed','Property_Area']
classification_model(model, df,predictor_var,outcome_var)

#Accuracy : 80.945% Cross-Validation Score : 80.946%

#Generally we expect the accuracy to increase on adding variables. But this is a more challenging case. The accuracy and cross-validation score are not getting impacted by less important variables. Credit_History is dominating the mode. We have two options now:

#Feature Engineering: dereive new information and try to predict those. I will leave this to your creativity.

#Better modeling techniques. Let’s explore this next.
 

#Decision Tree

#Decision tree is another method for making a predictive model. It is known to provide higher accuracy than logistic regression model. Read more about Decision Trees.

model = DecisionTreeClassifier()
predictor_var = ['Credit_History','Gender','Married','Education']
classification_model(model, df,predictor_var,outcome_var)

#Accuracy : 81.930% Cross-Validation Score : 76.656%

#Here the model based on categorical variables is unable to have an impact because Credit History is dominating over them. Let’s try a few numerical variables:

#We can try different combination of variables:
predictor_var = ['Credit_History','Loan_Amount_Term','LoanAmount_log']
classification_model(model, df,predictor_var,outcome_var)

#Accuracy : 92.345% Cross-Validation Score : 71.009%

#Here we observed that although the accuracy went up on adding variables, the cross-validation error went down. This is the result of model over-fitting the data. Let’s try an even more sophisticated algorithm and see if it helps:

 

#Random Forest

#Random forest is another algorithm for solving the classification problem. Read more about Random Forest.

#An advantage with Random Forest is that we can make it work with all the features and it returns a feature importance matrix which can be used to select features.

model = RandomForestClassifier(n_estimators=100)
predictor_var = ['Gender', 'Married', 'Dependents', 'Education',
       'Self_Employed', 'Loan_Amount_Term', 'Credit_History', 'Property_Area',
        'LoanAmount_log','TotalIncome_log']
classification_model(model, df,predictor_var,outcome_var)
#Accuracy : 100.000% Cross-Validation Score : 78.179%

#Here we see that the accuracy is 100% for the training set. This is the ultimate case of overfitting and can be resolved in two ways:

#Reducing the number of predictors
#Tuning the model parameters
#Let’s try both of these. First we see the feature importance matrix from which we’ll take the most important features.

#Create a series with feature importances:
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print featimp  

#let’s use the top 5 variables for creating a model. Also, we will modify the parameters of random forest model a little bit:

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['TotalIncome_log','LoanAmount_log','Credit_History','Dependents','Property_Area']
classification_model(model, df,predictor_var,outcome_var)
#Accuracy : 82.899% Cross-Validation Score : 81.461%

#Notice that although accuracy reduced, but the cross-validation score is improving showing that the model is generalizing well. Remember that random forest models are not exactly repeatable. Different runs will result in slight variations because of randomization. But the output should stay in the ballpark.

#You would have noticed that even after some basic parameter tuning on random forest, we have reached a cross-validation accuracy only slightly better than the original logistic regression model. This exercise gives us some very interesting and unique learning:

#Using a more sophisticated model does not guarantee better results.
#Avoid using complex modeling techniques as a black box without understanding the underlying concepts. Doing so would increase the tendency of overfitting thus making your models less interpretable
#Feature Engineering is the key to success. Everyone can use an Xgboost models but the real art and creativity lies in enhancing your features to better suit the model.
#So are you ready to take on the challenge? Start your data science journey with Loan Prediction Problem.

 

