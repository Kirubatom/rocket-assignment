#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_excel("C:/Users/senth/Downloads/Flight .xlsx")


# # Understanding the data

# In[3]:


data.head()


# In[4]:


data.tail()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.columns


# In[8]:


data.nunique()


# # Cleaning the data

# In[9]:


data.isnull().sum()


# In[10]:


data.Length.fillna(data.Length.median(),inplace = True)
data.Length.isnull().sum()


# In[11]:


data.PaperQuality.fillna('others',inplace = True)
data.PaperQuality.isnull().sum()


# In[12]:


data.Angle.fillna('others',inplace = True)
data.Angle.isnull().sum()


# In[13]:


#checking for outlier
plt.boxplot(data.Length)


# In[14]:


q3 = data.Length.quantile(0.75)
q1 = data.Length.quantile(0.25)
iqr=q3-q1
print(iqr)


# In[17]:


ue=q3+1.5*(iqr)
print(ue)
le=q1-1.5*(iqr)
print(le)


# In[18]:


data[(data.Length<ue)&(data.Length>le)]


# In[19]:


data.Length[data.Length>ue]=ue
data.Length[data.Length<le]=le


# In[20]:


plt.boxplot(data.Length)


# In[21]:


sns.countplot('Person',data=data)


# In[22]:


plt.xticks(rotation='vertical')
sns.countplot('Item_Identifier',data=data)


# In[23]:


plt.boxplot(data.Breadth)


# In[24]:


sns.countplot('Dominanthand',data=data)


# In[25]:


plt.boxplot(data.Distance)


# In[27]:


q3=data.Distance.quantile(0.75)
q1=data.Distance.quantile(0.25)
iqr=q3-q1
print(iqr)


# In[28]:


ue=q3+1.5*(iqr)
print(ue)
le=q1-1.5*(iqr)
print(le)


# In[29]:


data[(data.Distance<ue)&(data.Distance>le)]


# In[30]:


data.Distance[data.Distance>ue]=ue
data.Distance[data.Distance<le]=le


# In[31]:


plt.boxplot(data.Distance)


# In[32]:


sns.countplot('PaperQuality',data=data)


# In[33]:


sns.countplot('Angle',data=data)


# In[34]:


pd.get_dummies(data.Person)


# In[35]:


dummy = pd.get_dummies(data)
dummy


# In[36]:


x=dummy.drop('Distance',axis=1)
y=dummy.Distance


# In[38]:


from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.2)


# In[42]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()


# In[44]:


model.fit(x_train,y_train)


# In[45]:


model.predict(x_test)


# In[46]:


predicted_y = model.predict(x_test)


# In[47]:


model.score(x_test,y_test)


# In[48]:


from sklearn.metrics import mean_squared_error


# In[50]:


mean_squared_error(y_test,predicted_y)


# In[52]:


import numpy as np
data['Dist1']=np.where(data['Distance']>5,True,False)
data


# In[53]:


pd.get_dummies(data.Person)


# In[54]:


dummy=pd.get_dummies(data)
dummy


# In[55]:


x=dummy.drop('Dist1',axis=1)
y=dummy.Dist1


# In[57]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=25)


# In[58]:


from sklearn.linear_model import LogisticRegression
model=LogisticRegression()


# In[59]:


model.fit(x_train,y_train)


# In[60]:


model.predict(x_test)


# In[61]:


predicted_y=model.predict(x_test)


# In[62]:


model.score(x_test,y_test)


# In[64]:


probability=model.predict_proba(x_test)[:,1]
probability


# In[65]:


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predicted_y)
cm


# In[ ]:




