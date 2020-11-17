#!/usr/bin/env python
# coding: utf-8

# # House Price Prediction

# In[2]:


# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


#reading the data using pandas.pd.read_csv()method creates a DataFrame from a csv file
#load dataset
train = pd.read_csv('./train.csv')


# In[4]:


#showing the first five rows of the dataset
train.head()


# In[5]:


#showing the last five rows of the dataset
train.tail()


# In[6]:


#shape of train data
#Showing the number of rows and columns in the dataset
train.shape


# In[7]:


train.info()


# In[9]:


#now so to get the target variable we need the testing data set 
#reading the test dataset using the pandas
test = pd.read_csv('./test.csv')


# In[10]:


#To visualize the train data and view its distribution using graph
#some analysis on target variable
plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#As we can see from the above graph data is not normalized properly 
#so we need to do normalize for better data distribution
#this target varibale is right skewed. now, we need to tranform this variable and make it normal distribution.


# In[11]:


#Here we use log for target variable to make more normal distribution
#we use log function which is in numpy
train['SalePrice'] = np.log1p(train['SalePrice'])

#Check again for more normal distribution

plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)

# Get the fitted parameters used by the function

(mu, sigma) = stats.norm.fit(train['SalePrice'])

# plot with the distribution

plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')

#Probablity plot

fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[ ]:


#Now we can see data is distributed in much better way
#By using the log function we were able to normalize the data


# In[12]:


#Check Missing values
#Let's check if the data set has any missing values. 
train.columns[train.isnull().any()]


# In[13]:


#plot of missing value attributes
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()


# In[ ]:


#As the above heatmap shows the missing values
#the white spaces on the heatmap are the missing values
#we need to fill those missing values to get the accurate result


# In[14]:


#to get rid of this missing values we need to count them
#missing value counts in each of these columns
#the below code will give the detail info about number of missing value of particular columns
Isnull = train.isnull().sum()/len(train)*100
Isnull = Isnull[Isnull>0]
Isnull.sort_values(inplace=True, ascending=False)
Isnull


# In[15]:


#to visualize the missing values
#we need to convert missing values into dataframe
Isnull = Isnull.to_frame()


# In[16]:


Isnull.columns = ['count']


# In[17]:


Isnull.index.names = ['Name']


# In[18]:


Isnull['Name'] = Isnull.index


# In[19]:


#plot Missing values 
#ploting this missing values will give better look at the missing data 
plt.figure(figsize=(13, 5))
sns.set(style='whitegrid')
sns.barplot(x='Name', y='count', data=Isnull)
plt.xticks(rotation = 90)
plt.show()


# In[ ]:


#here as we can see PoolQC and MiscFeature have the highest missing values
#where as MasVnrType and Electrical have the least missing values


# In[23]:


#Now we are going to show correlation between train attributes
#Separate variable into new dataframe from original dataframe which has only numerical values
#there is 38 numerical attribute from 81 attributes
train_corr = train.select_dtypes(include=[np.number])


# In[24]:


#for showing the correlated dataset's rows and columns
train_corr.shape


# In[25]:


#Delete Id because that is not need for corralation plot
del train_corr['Id']


# In[26]:


#Correlation plot
#this will give detail visual look about how the columns are correlated with each other
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)


# In[ ]:


#From the above correlation plot we can see that fields with lighter color or
# fields with higher values(from 0 to 1) are highly correalated and 
# the ones with darker field or lesser values are lesser correlated


# In[27]:


#Top 50% Correlation train attributes with sale-price
#this will give the columns which are correlated with respect to the Sale price i.e. more than 0.5 or 50 %
top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# In[ ]:


#As from the above correlation plot we can see that 
#Here TotRmsAbvgrd is least correlated with target feature of saleprice by 53% 
#Here OverallQual is highly correlated with target feature of saleprice by 82%


# In[31]:


#unique value of OverallQual
train.OverallQual.unique()


# In[32]:


sns.barplot(train.OverallQual, train.SalePrice)


# In[ ]:


#the above graph shows the effects or relation of unique values of OverallQual on the saleprice 
#we can see that higher the unique value of OverallQual higher is the saleprice for that unique saleprice


# In[33]:


#Plotting a boxplot to show relation between OverallQual and Saleprice
#A box plot (or box-and-whisker plot) shows the distribution of quantitative data in a way that 
#facilitates comparisons between variables or across levels of a categorical variable. 
#The box shows the quartiles of the dataset while the whiskers extend to show the rest of the distribution, 
#except for points that are determined to be “outliers” using a method that is a function of the inter-quartile range.
plt.figure(figsize=(18, 8))
sns.boxplot(x=train.OverallQual, y=train.SalePrice)


# In[ ]:


#the above boxplot shows the top and bottom quartile and using the box 
#the outliers are shown using dotted line and 
# whiskers using the lines


# In[34]:


#Ploting pairplot to show relation between diffrent columns
#Plot pairwise relationships in a dataset.
#By default, this function will create a grid of Axes such that 
#each variable in data will by shared in the y-axis across a single row and 
#in the x-axis across a single column. The diagonal Axes are treated differently, 
#drawing a plot to show the univariate distribution of the data for the variable in that column.
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.set(style='ticks')
sns.pairplot(train[col], height=3, kind='reg')


# In[ ]:


#The pairs plot builds on two basic figures, the histogram and the scatter plot. 
#The histogram on the diagonal allows us to see the distribution of a single variable 
#while the scatter plots on the upper and lower triangles show the relationship between two variables.


# In[35]:


#this will give features that are related to target i.e. saleprice in descending order 
print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice


# In[36]:


#Now we are going to fill the missing value from each columns with missing values
# PoolQC has missing value ratio is 99%+. So, there is fill by None
train['PoolQC'] = train['PoolQC'].fillna('None')


# In[37]:


#Around 50% missing values attributes have been fill by None
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')


# In[38]:


#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))


# In[39]:


#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')


# In[40]:


#GarageYrBlt, GarageArea and GarageCars these are replacing with zero
for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))


# In[41]:


#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')


# In[42]:


#MasVnrArea : replace with zero
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))


# In[43]:


#MasVnrType : replace with None
train['MasVnrType'] = train['MasVnrType'].fillna('None')


# In[44]:


#There is put mode value 
train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]


# In[45]:


#There is no need of Utilities
train = train.drop(['Utilities'], axis=1)


# In[47]:


#Checking there is any null value or not
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()


# In[ ]:


#from the above graph we can see that there are no missing values 
#all the missing values are filed by None or zero


# In[48]:


#Encoding str to int
cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')


# In[49]:


from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))


# In[50]:


#for preparing the data for prediction
#Take target variable into y
#storing the saleprice into a variable y
y = train['SalePrice']


# In[51]:


#As the saleprice price is stored in a variable y
#Delete the saleprice
del train['SalePrice']


# In[52]:


#Take their values in X and y
#store the values of train and y(saleprice) in variables x and y respectively
X = train.values
y = y.values


# In[53]:


# Split data into train and test format
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[54]:


#Linear Regression
#The objective of a linear regression model is to find a relationship between one or more features(independent variables)
#and a continuous target variable(dependent variable).
#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()


# In[55]:


#Fitting the model
model.fit(X_train, y_train)


# In[56]:


#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))


# In[57]:


#Checking the Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[58]:


#Random Forest Regression
#A Random Forest is an ensemble technique capable of performing both regression and classification tasks
#with the use of multiple decision trees and a technique called Bootstrap Aggregation, commonly known as bagging.
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)


# In[59]:


#Fitting the model 
model.fit(X_train, y_train)


# In[60]:


#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))


# In[61]:


#Checking the Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[68]:


#Gradient Boosting Regression
#Boosting is a sequential technique which works on the principle of ensemble. 
#It combines a set of weak learners and delivers improved prediction accuracy.
#At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. 
#The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher.
#Train the model
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor(n_estimators=100, max_depth=4)


# In[69]:


#Fitting the model
model.fit(X_train, y_train)


# In[70]:


#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))


# In[72]:


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)

