#!/usr/bin/env python
# coding: utf-8

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[83]:


# Reading data from remote link
url = "http://bit.ly/w-data"
df = pd.read_csv(url)
print("Data imported successfully")

df.head(100)


# Let's plot our data points on 2-D graph to eyeball our dataset and see if we can manually find any relationship between the data. We can create the plot with the following script:

# In[84]:


# Plotting the distribution of scores
df.plot(x='Hours', y='Scores', style='o')
plt.title('Hours vs Percentage')
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()


# In[85]:


df.isnull().sum()


# **From the graph above, we can clearly see that there is a positive linear relation between the number of hours studied and percentage of score.**

# In[86]:


sns.countplot(x='Scores',data=df)


# In[87]:


df.describe()


# In[88]:


df.groupby('Hours')['Scores'].agg(['count', 'mean'])


# In[89]:


sns.boxplot(x='Hours',y='Scores',data=df)


# In[90]:


sns.heatmap(df.isnull(),cbar=False,cmap='viridis')


# ### **Preparing the data**
# 
# The next step is to divide the data into "attributes" (inputs) and "labels" (outputs).

# In[91]:


X = df.iloc[:, :-1].values
y = df.iloc[:, 1].values


# Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

# In[92]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)


# ### **Training the Algorithm**
# We have split our data into training and testing sets, and now is finally the time to train our algorithm.

# In[93]:


from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(max_depth=100)
RFC.fit(X_train, y_train)


# ### **Making Predictions**
# Now that we have trained our algorithm, it's time to make some predictions.

# In[94]:


y_pred = RFC.predict(X_test)


# In[95]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df


# ### **Evaluating the model**
# 
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset. For simplicity here, we have chosen the mean square error. There are many such metrics.

# In[96]:


from sklearn import metrics
print('Mean Absolute Error:',
      metrics.mean_absolute_error(y_test, y_pred))

