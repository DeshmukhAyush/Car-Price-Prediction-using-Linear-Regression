# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 23:03:32 2023

@author: AYUSH DESHMUKH
"""

# Dataset : https://www.kaggle.com

# Importing libraries
# We are importing pandas Library as pd
import pandas as pd

# Import train_test_split
from sklearn.model_selection import train_test_split

# We can import Linear Regression 
from sklearn.linear_model import LinearRegression

# We are importing numpy as np
import numpy as np

# Importing the dataset from the local storage
df = pd.read_csv("F:/Car Price Prediction.csv")

# Dividing the dataset into Dependent variables and Independent variables.
# X is the independent variable
# Y is the dependent variable

x=df[['Year', 'Mileage']]
y=df['Price']

# Splitting the dataset into training and testing datasets
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0)

# Making LSR as an object of LinearRegression
LSR=LinearRegression()

# Fitting the LSR with the training datasets
LSR.fit(x_train.values, y_train.values)

# We are predicting the output on the basis of the value which is passed in the predict function 
predicted_values = LSR.predict(x_test)

# Creating the required desired values
data = np.array([[2015, 20]])

# Predicting the output on the basis of numpy array which is passes to predict function
predict=LSR.predict(data)

# Printing the output
print("Predicted price is", predict)

