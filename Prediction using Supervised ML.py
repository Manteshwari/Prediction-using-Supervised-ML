#!/usr/bin/env python
# coding: utf-8

# # GRIP - The Sparks Foundation
# 
# ## Data Science and Business Analytics Intern
# 
# ## Author : Manteshwari Lomeshwar Pipare
# 
# ## TASK 1 : Prediction using Supervised ML
# 
# ### This is the task to Predict the percentage of marks of the students based on the number of hours they studied. This is a simple linear regression task as it involving two features. 

# In[107]:


#importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics


# In[108]:


dataset = pd.read_csv("http://bit.ly/w-data")
print("Data imported successfully")
dataset


# In[109]:


#Checking if any missing value 
print("\nMissing Value : ",dataset.isnull().sum().values.sum())


# In[110]:


#Dataset description
dataset.describe()


# In[111]:


X = dataset.iloc[:, :-1].values
#print(X)
X


# In[112]:


Y = dataset.iloc[:,1].values
#print(Y)
Y


# # visualizing the scores vs hours data

# In[113]:


dataset.plot(x='Hours',y='Scores',style='o')
plt.title('Scores vs Hours')
plt.ylabel('Scores Percentage')
plt.xlabel('Hours Studied')
plt.show()


# # Train-Test Split

# In[114]:


# Spliting the Data in two
train_X, val_X, train_Y, val_Y = train_test_split(X, Y)


# In[115]:


regressor = LinearRegression()  
regressor.fit(train_X, train_Y) 


# # Plotting the regression line

# In[116]:


line = regressor.coef_*X+regressor.intercept_

# Plotting for the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()


# # Predictions of Marks

# In[117]:


pred_y = lr.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[118]:


df = pd.DataFrame({'Actual': val_Y, 'Predicted': pred_y})  
df


# ### Visually Comparing the Predicted Marks with the Actual Marks

# In[121]:


plt.scatter(x=val_X, y=val_Y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted')
plt.ylabel('Marks Percentage')
plt.xlabel('Hours Studied')
plt.show()


# In[119]:


print('Mean Absolute Error:', 
      metrics.mean_absolute_error(val_Y, pred_y))


# In[122]:



hours = [9.25]
answer = lr.predict([hours])
print(answer)


# ### if a student study 9.25 hours then He/She will score 94.17

# In[ ]:


thank you

