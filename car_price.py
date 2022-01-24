#!/usr/bin/env python
# coding: utf-8

# ### Importing the dependencies

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# ### Data collection and Processing

# In[2]:


# loading the data from csv file to pandas dataframe

car_dataset = pd.read_csv('car data.csv')


# In[3]:


# inspecting the first 5 rows of the dataframe

car_dataset.head()


# In[4]:


# checking the number of rows and columns

car_dataset.shape


# In[5]:


# getting some information about the dataset

car_dataset.info()


# In[6]:


# checking the number of missing values

car_dataset.isnull().sum()


# In[7]:


# checking the distribution of categorical data

print(car_dataset.Fuel_Type.value_counts())
print(car_dataset.Seller_Type.value_counts())
print(car_dataset.Transmission.value_counts())


# ### Encoding the categorical data

# In[8]:


# encoding "Fuel_Type" Column
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel':1, 'CNG': 2}}, inplace= True)

# encoding "Seller_Type" Column
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace= True)

# encoding "Transmission" Column
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace= True)


# In[9]:


car_dataset.head()


# ### Splitting the features and labels

# In[10]:


X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis = 1)
Y = car_dataset['Selling_Price']


# In[11]:


X


# In[12]:


Y


# ### Splitting the data into training and test data

# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 2)


# ### Model Training

# ### 1. Linear Regression

# In[14]:


# loading the linear regresion model

lin_model = LinearRegression()


# In[15]:


lin_model.fit(X_train, Y_train)


# ### Evaluating Model

# In[16]:


# prediction on training data

prediction_train = lin_model.predict(X_train)


# In[20]:


# R-squared value

rsq_train = metrics.r2_score(Y_train, prediction_train)
print('R-squared value: ', rsq_train)


# ### Visualize the actual prices and predicted prices

# In[21]:


plt.scatter(Y_train, prediction_train)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs. Predicted Prices')
plt.xlim([-7, 35])
plt.ylim([-7, 35])
plt.show()


# In[22]:


# prediction on test data

prediction_test = lin_model.predict(X_test)


# In[23]:


# R-squared value

rsq_test = metrics.r2_score(Y_test, prediction_test)
print('R-squared value: ', rsq_test)


# In[24]:


plt.scatter(Y_test, prediction_test)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs. Predicted Prices')
plt.xlim([-5, 13])
plt.ylim([-5, 13])
plt.show()


# ### 2. Lasso Regression

# In[25]:


# loading the lasso regression model

lass_model = Lasso()


# In[26]:


lass_model.fit(X_train, Y_train)


# ### Evaluating model

# In[27]:


# prediction on training data

pred_train = lass_model.predict(X_train)


# In[28]:


# R-squared value

r_train = metrics.r2_score(Y_train, pred_train)
print('R-squared value: ', r_train)


# ### Visualize the actual prices and predicted prices

# In[29]:


plt.scatter(Y_train, pred_train)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs. Predicted Prices')
plt.xlim([-7, 35])
plt.ylim([-7, 35])
plt.show()


# In[30]:


# prediction on test data

pred_test = lass_model.predict(X_test)


# In[31]:


# R-squared value

r_test = metrics.r2_score(Y_test, pred_test)
print('R-squared value: ', r_test)


# In[32]:


plt.scatter(Y_test, pred_test)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual Prices vs. Predicted Prices')
plt.xlim([-5, 13])
plt.ylim([-5, 13])
plt.show()


# ### Saving the model

# Save Model

# In[33]:


with open('car_price.pkl', 'wb') as f:
    pickle.dump(lass_model, f)


# Load Model

# In[34]:


with open('car_price.pkl', 'rb') as f:
    loaded_lass_model = pickle.load(f)

