#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries


# In[1]:


import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


import nltk
nltk.download('stopwords')


# In[3]:


print(stopwords.words('english'))


# In[ ]:


#DATA COLLECTION AND PREPROCESSING


# In[4]:


#twitter data as td
td=pd.read_csv('training.1600000.processed.noemoticon.csv', encoding = 'ISO-8859-1')


# In[5]:


td.shape


# In[6]:


td.head(100)


# In[7]:


column_names = ['target','id','date','flag','user','text']
td=pd.read_csv('training.1600000.processed.noemoticon.csv', names=column_names,encoding = 'ISO-8859-1')


# In[9]:


print(td.shape)
print(td.head())
print(td.isnull().sum())


# In[9]:


td1=td.head(100000)


# In[10]:


td1.shape


# In[11]:


td1['target'].value_counts()


# In[12]:


td1.shape


# In[ ]:


#0--NEGATIVE TWEET
#1--POSITIVE TWEET


# In[13]:


port_stem=PorterStemmer()


# In[14]:


def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


# In[15]:


td1['stemmed_content'] = td1['text'].apply(stemming)


# In[16]:


td1.head()


# In[17]:


print(td1['stemmed_content'])


# In[18]:


X = td1['stemmed_content'].values
Y = td1['target'].values


# In[19]:


print(X)


# In[20]:


print(Y)


# In[43]:


#SPLITTING THE DATA IN TO TRAINING AND TESTING


# In[21]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,stratify=Y,random_state=2)


# In[22]:


print(X.shape,X_train.shape,X_test.shape)


# In[23]:


print(X_train)


# In[24]:


print(X_test)


# In[25]:


vectorizer=TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)


# In[26]:


print(X_train)


# In[27]:


print(X_test)


# In[44]:


#TRAING MACHINE MODEL


# In[45]:


#LOGISTIC REGRESSION


# In[28]:


model = LogisticRegression(max_iter=10000)


# In[29]:


model.fit(X_train, Y_train)


# In[46]:


#MODEL EVAULATION


# In[30]:


X_train_prediction = model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train, X_train_prediction)


# In[31]:


print('Acuuracy score on training data:', training_data_accuracy)


# In[32]:


X_test_prediction = model.predict(X_test)
test_data_accuracy=accuracy_score(Y_test, X_test_prediction)


# In[33]:


print('Acuuracy score on training data:', test_data_accuracy)


# In[35]:


#saving trainig data


# In[34]:


import pickle


# In[36]:


filename = 'trained_model.sav'
pickle.dump(model, open(filename, 'wb'))


# In[37]:


#using saved model for future prediction


# In[39]:


loaded_model = pickle.load(open('trained_model.sav', 'rb'))


# In[41]:


X_new = X_test[200]
print(Y_test[200])

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print('Negative Tweet')
else:
    print('Positive Tweet')


# In[42]:


X_new = X_test[3]
print(Y_test[3])

prediction = model.predict(X_new)
print(prediction)

if (prediction[0]==0):
    print('Negative Tweet')
else:
    print('Positive Tweet')


# In[ ]:




