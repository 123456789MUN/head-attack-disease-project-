#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


df=pd.read_csv("dataset.csv")


# In[5]:


df


# In[9]:


df.head()


# In[10]:


df.tail()


# In[13]:


df.loc


# In[7]:


df.iloc[1]


# In[17]:


df.describe()


# In[18]:


df.info()


# In[16]:


# Count the number of patients with and without heart disease
heart_disease = df[df['target'] == 1].shape[0]
no_heart_disease = df[df['target'] == 0].shape[0]

# Creating the chart
labels = ['Heart Disease', 'No Heart Disease']
values = [heart_disease, no_heart_disease]

plt.bar(labels, values)
plt.title('Number of Patients with/without Heart Disease')
plt.xlabel('Disease Status')
plt.ylabel('Number of Patients')

# Display the chart
plt.show()


# In[20]:


#plt.bar(labels, values)
plt.bar([df[df['target']==1]['age'].shape(0),df[df['target']==0]['age']],labels=["No Heart Disease","Heart Disease"])
plt.title('age vs Disease Status')
plt.xlabel('Disease Status')
plt.ylabel('age')

# Display the chart
plt.show()


# In[ ]:


b. Visualize the age and whether a patient has disease or not


# In[22]:


# Create a boxplot to visualize age and disease status
plt.figure(figsize=(10, 6))
plt.boxplot([df[df['target'] == 0]['age'], df[df['target'] == 1]['age']], labels=['No Heart Disease', 'Heart Disease'])
plt.title('Age vs Disease Status')
plt.xlabel('Disease Status')
plt.ylabel('Age')

# Display the chart
plt.show()


# In[ ]:


#c. Visualize correlation between all features using a heat map


# In[29]:


import seaborn as sns


# In[30]:


correlation_matrix = df.corr()

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu")

# Show the plot
plt.title('Correlation Heatmap')
plt.show()


# In[ ]:


i. Divide the dataset in 70:30 ratio


# In[32]:


from sklearn.model_selection import train_test_split


# In[33]:


# Split the dataset into training and testing sets
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the training and testing sets
print("Training set shape:", X_train.shape, y_train.shape)
print("Testing set shape:", X_test.shape, y_test.shape)


# In[ ]:


#i. Build the model on train set and predict the values on test set


# In[35]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[36]:


# Create a logistic regression model
model = LogisticRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Predict the target variable on the testing set
y_pred = model.predict(X_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:


#iii. Build the confusion matrix and get the accuracy score


# In[37]:


from sklearn.metrics import confusion_matrix, accuracy_score

# Build the confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:




