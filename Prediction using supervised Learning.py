#!/usr/bin/env python
# coding: utf-8

# # The Sparks Foundation
# 
# 
# ## Task 1: Predicting the percentage of student based on the number of hours they study.
# 
# ## Author: Pratiksha Jadli

# Description:
# Data of study hours with corresponding scores has been provided in the below link
# Dataset: http://bit.ly/w-data
# 
# 
# Main Problem:
# Here, I want to predict the percentage of student based on the number of hours using the supervised learning algorithm.
# As in this case there is only one independent variable and one dependent variable and therefore, I 'm using the Simple Linear Regression technique.

# ##  Importing neccesary libraries like numpy, pandas, matplotlib, seaborn, sklearn

# In[97]:


import numpy as np  # For numerical analysis like using mathematical operations
import pandas as pd # For Data manipulation
import matplotlib.pyplot as plt # For Data visulization like charts/graphs
import seaborn as sns # For Data visulization, using sens.set() to map the seaborn themes to matplotlib 
sns.set()
from sklearn.linear_model import LinearRegression # Sklearn is for machine learning libraries 


# ## Load the data

# In[98]:


# Data file link stored in variable namely 'link'
link="http://bit.ly/w-data"

# Reading the data
data=pd.read_csv(link)
data


# ## Describing the data

# In[99]:


# Statistics information of the data
data.describe()


# In[100]:


# Looking for null values in the data, if any.
data.isnull().any()


# # Defing independent and dependent variables

# In[71]:


# Defining independent variable (Hours) using the iloc function
X=data.iloc[:,:1].values 

# Defining dependent variable (Scores) using the iloc function
y=data.iloc[:,1].values 


# In[101]:


# Plotting Hours Vs Scores
plt.scatter(X,y,c='red')
plt.xlabel("Hours",fontsize=20, c='orange')
plt.ylabel("Scores",fontsize=20, c='green')
plt.show()


# ### Clearly from the above graph, Hours and Scores holds a relation between them.

# # Linear Regression Model

# In simple linear regression model, two variables holds a causal relationship between them, i.e. change in one variable leads to change in other one as well.
# 
# Y = (b_0) + (b_1) * X where
# 
# X- Independent variable, 
# Y- Dependent variable, 
# b_0- Intercept, 
# b_1- Coefficient of independent variable
# 

# ### Dividing the original dataset into train and test data, with test size of 80%-20%, i.e. 80% train data and 20% test data, using the sklearn, train-test split.

# In[102]:



from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# ### Defining the model

# In[103]:



reg=LinearRegression()

# Firstly, training the Linear regression model with train data. 
reg.fit(X_train,y_train)


# In[104]:


reg.coef_ # Coefficient of independent variable(b_1)


# In[105]:


reg.intercept_ # Intercept (b_0)


# # Making Prediction

# ### Testing the model by passing the test data

# In[106]:



pred=reg.predict(X_test)


# ### New data frame has been ceated to show the actual and predicted scores, corresponding to the hours test data.

# In[107]:



A_P={"Actual(Scores)": y_test ,"Predicted(Scores)": pred}
new_data=pd.DataFrame(A_P)
new_data["X_test(Hours)"]= X_test
new_data.set_index('X_test(Hours)')


# ### Fitting regression line 

# In[108]:



plt.scatter(X,y,c='blue')
y_hat1=reg.intercept_+reg.coef_*X
plt.plot(X,y_hat1,lw=4,c='red',label= 'Regression Line')
plt.xlabel("Hours",fontsize=20,c='orange')
plt.ylabel("Scores",fontsize=20,c='green')
plt.legend()
plt.show()


# ## Predict the score of student who has been studying for 9.5 hours.

# In[109]:


pred_val=np.array([9.5])
y_hat1=reg.intercept_+reg.coef_*pred_val
y_hat1


# ## Model Evaluation Metric

# ### To quantify the model performance, mean absolute error and mean squared error are used in this case.

# In[95]:




from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt as sq


# In[110]:


print("MAE test score : ", mean_absolute_error(y_test,pred))
print("RMSE test score : ", sq(mean_squared_error(y_test,pred)))


# In[ ]:




