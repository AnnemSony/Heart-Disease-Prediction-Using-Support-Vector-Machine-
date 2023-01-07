#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree


# In[7]:


data=("C:/Users/Lenovo/Desktop/heart.csv")
df=pd.read_csv(data, sep=',', header=0)


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(data)


# In[9]:


X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109) # 70% training and 30% test


# In[22]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=12000)
model.fit(X_train, y_train)
y_pred=model.predict(X_test)


# In[23]:


from sklearn import metrics
print("Acuracy Using LOGISTIC REGREESION",metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


from sklearn.naive_bayes import GaussainNB
gnb=GaussainNB()
gnb.fit(X_train, y_train)
y_pred=gnb.predict(X_test)


# In[27]:


print("Acuuracy using Bayesian",metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:


from sklearn.tree import DecisionTreeClassifier
dtree=DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None)
dtree.fit(X_train, y_train)
y_pred=dtree.predict(X_test)


# In[28]:


print("Acuraccy using DECISIONTREE:", metrics.accuracy_score(y_test, y_pred)*100)


# In[29]:


from sklearn.ensemble import RandomForestClassifier
rfm=RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=1)
rfm.fit(X_train, y_train)
y_pred=rfm.predict(X_test)


# In[30]:


print("Acuraccy using RANDOMFOREST:", metrics.accuracy_score(y_test, y_pred)*100)


# In[ ]:





# In[33]:


from sklearn.neighbors import KNeighborsClassifier     
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
y_pred_knn=knn.predict(X_test)


# In[34]:


print("Acuraccy using KNeighborsClassifier:", metrics.accuracy_score(y_test, y_pred)*100)


# In[25]:


from sklearn.svm import SVC
svm=SVC(kernel="linear", C=0.025, random_state=101)
svm.fit(X_train, y_train)
y_pred=svm.predict(X_test)


# In[26]:


from sklearn import metrics
print("Acuracy Using SUPPORT VECTOR MACHINE",metrics.accuracy_score(y_test, y_pred)*100)


# In[46]:


x=["LOGISTIC", "DECISIONTREE", "RANDOMFOREST", "SVM", "BAYES", "KNN"]
y=[87, 89, 89, 89, 89, 89]
sns.set(rc={'figure.figsize':(15,8)})
sns.barplot(x,y)


# In[44]:


sns.barplot(x,y)


# In[ ]:




