#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION- DATA SCIENCE AND BUSINESS ANALYTICS
# 

# # TASK 1

# DONE BY- VAISHNAVI SHEORAN

# # Problem Statement

# *Predict the percentage of an student based on the no. of study hours.
# 
# *This is a simple linear regression task as it involves just 2 variables.
# 
# *What will be predicted score if a student studies for 9.25 hrs/ day?
# 
# 

# In[2]:


# Importing essential libraries
import pandas as pd
import numpy as np  
import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df


# In[5]:


#random 10 records

df.head(10)


# In[6]:


#last 5 records

df.tail()


# In[7]:


#let's see percentiles,mean,std,max,count of the given dataset.
df.describe()


# In[8]:


#Let's print the full summary of the dataframe 

df.info()


# In[9]:


plt.boxplot(df)
plt.show()


# # VISUALIZING DATA

# In[10]:


# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[11]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values  


# In[12]:


X


# In[13]:


Y


# # Preparing Data and splitting into train and test sets.

# In[14]:


from sklearn.model_selection import train_test_split  
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) 


# In[15]:


print("X train.shape =", X_train.shape)
print("Y train.shape =", Y_train.shape)
print("X test.shape  =", X_test.shape)
print("Y test.shape  =", Y_test.shape)


# # TRAINING THE MODEL

# In[16]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train) 

print("Training complete.")


# In[17]:


# Plotting the regression line
line = regressor.coef_*X+regressor.intercept_


# In[18]:


# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line, color='red');
plt.title("Regression line(Train set)",fontsize=10)
plt.xlabel("Hours",fontsize=15)
plt.ylabel("Scores",fontsize=15)
plt.show()


# # TESTING THE DATA

# In[19]:


print(X_test) # Testing data - In Hours
y_pred = regressor.predict(X_test) # Predicting the scores


# In[20]:


# Comparing Actual vs Predicted
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})  
df 


# # ACCURACY OF THE MODEL

# In[21]:


from sklearn import metrics
metrics.r2_score(Y_test,y_pred) ##Goodness of fit Test


# above 94% percentage indidicates that above fitted model is a GOOD MODEL

# # Predicting the Error

# In[22]:


from sklearn.metrics import mean_squared_error,mean_absolute_error
MSE = metrics.mean_squared_error(Y_test,y_pred)

root_E = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
Abs_E = np.sqrt(metrics.mean_squared_error(Y_test,y_pred))
print("Mean Squared Error      = ",MSE)
print("Root Mean Squared Error = ",root_E)
print("Mean Absolute Error     = ",Abs_E)


# In[23]:


# Testing with own data
hours = 9.25
own_pred = regressor.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# # CONCLUSION

# From the above result we can say that if a student studied for 9.25 hours then student will secure 93,69 marks

# In[ ]:




