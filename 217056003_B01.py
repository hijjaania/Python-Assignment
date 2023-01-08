#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""Tumor data from UCI Machine Learning"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from array import array
from statistics import mean, stdev
from math import sqrt
import sys


# In[6]:


data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/primary-tumor/primary-tumor.data", header=None,
                  names=['Class', 'Age', 'Sex', 'Histologic Type', 'Degree of Differentiation', 'Bone', 'Bone Marrow', 'Lung', 'Pleura',
                        'Peritoneum', 'Liver', 'Brain', 'Skin', 'Neck', 'Supraclavicular', 'Axillar', 'Mediastium', 'Abdominal'])
df = pd.DataFrame(data)


# In[7]:


#Replace with NaN (Not a Number)
def replace_missing(df):
    for i, j in df.iterrows():
        ct = 0
        for col in j:
            if(col == '?'):
                df.iloc[i,ct] = np.nan
            ct += 1
    return df


# In[70]:


df_new = replace_missing(df)
df_new


# In[84]:


#Choose some attributes and display as Bar Charts
age_col = df['Age']
age_data = {}
for x in age_col:
    age_data[x] = age_data.get(x, 0) + 1
ages = []
freq = []
for i in age_data:
    ages.append(i)
    freq.append(age_data.get(i))
x = ages
y = freq
plt.bar(x,y,align='center')
plt.xlabel('Age')
plt.ylabel('Number of Cases')
for i in range(len(y)):
    plt.hlines(y[i],0,x[i], color="red",
              linestyle=':')
plt.show()


# In[75]:


#Choose some attributes and display as Bar Charts
lung_col = df['Lung']
lung_data = {}
for x in lung_col:
    lung_data[x] = lung_data.get(x, 0) + 1
lung = []
freq = []
for i in lung_data:
    lung.append(i)
    freq.append(lung_data.get(i))
x = lung
y = freq
plt.bar(x,y,align='center')
plt.xlabel('Lung')
plt.ylabel('Number of Cases')
for i in range(len(y)):
    plt.hlines(y[i],0,x[i], color="red",
              linestyle=':')
plt.show()


# In[169]:


#Choose some attributes and display as Histogram
plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=np_hist, bins=8, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Age', fontsize=15)
plt.ylabel('Number of Cases', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Number of Cases', fontsize=15)
plt.title('Histogram of Age in Tumor Primary Data')
plt.show()


# In[170]:


#Choose some attributes and display as Histogram
plt.figure(figsize=[10,8])
n, bins, patches = plt.hist(x=np_hist, bins=5, color='#0504aa', alpha=0.7, rwidth=0.85)
plt.grid(axis='y', alpha=1)
plt.xlabel('Lung', fontsize=15)
plt.ylabel('Number of Cases', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('Number of Cases', fontsize=15)
plt.title('Lung Tumor Distribution Histogram')
plt.show()


# In[8]:


#Check if there are outliers in Age Column
age_col = df['Age']
age_data = {}
for x in age_col:
    age_data[x] = age_data.get(x, 0)+1
dict(list(age_data.items()))


# In[9]:


#Summary Statistics
#Mean of Age Distribution
np.array(age_col).mean()


# In[10]:


#Summary Statistics
#Variance of Age Distribution
np.array(age_col).var()


# In[11]:


#Summary Statistics
#Standard Deviation of Age Distribution
np.array(age_col).std()


# In[7]:


#Extract Age Element from Dictionary 
def get_dict_element(a_dict):
    arr = []
    for x in range(len(a_dict)):
        arr.append(a_dict.get(x, 1))
    return arr  


# In[17]:


#effect size between male and female ages
m = np.array(get_dict_element(age_col))
f = np.array(get_dict_element(age_col))
print(m[:5])
print(f[:5])


# In[18]:


#Calculate Effect Size
def CohenEffectSize(group1,group2):
    diff = group1.mean() - group2.mean()
    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)
    
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    
    return d


# In[19]:


#Calculate Effect Size
d = CohenEffectSize(m,f)
print('d = {}'.format(d))

#Very Small Effect Size


# In[20]:


#Check if there are outliers in Lung Column
lung_col = df['Lung']
lung_data = {}
for x in lung_col:
    lung_data[x] = lung_data.get(x, 0)+1
dict(list(lung_data.items()))


# In[21]:


#Summary Statistics
#Mean of Lung Tumor Distribution
np.array(lung_col).mean()


# In[22]:


#Summary Statistics
#Variance of Lung Tumor Distribution
np.array(lung_col).var()


# In[23]:


#Summary Statistics
#Standard Deviation of Lung Tumor Distribution
np.array(lung_col).std()


# In[24]:


#Extract Lung Element from Dictionary 
def get_dict_element(a_dict):
    arr = []
    for x in range(len(a_dict)):
        arr.append(a_dict.get(x, 1))
    return arr  


# In[25]:


#effect size between male and female in Lung Tumor Case
m = np.array(get_dict_element(lung_col))
f = np.array(get_dict_element(lung_col))
print(m[:5])
print(f[:5])


# In[26]:


#Calculate Effect Size
def CohenEffectSize(group1,group2):
    diff = group1.mean() - group2.mean()
    var1 = group1.var()
    var2 = group2.var()
    n1, n2 = len(group1), len(group2)
    
    pooled_var = (n1 * var1 + n2 * var2) / (n1 + n2)
    d = diff / math.sqrt(pooled_var)
    
    return d


# In[27]:


#Calculate Effect Size
d = CohenEffectSize(m,f)
print('d = {}'.format(d))

#Very Small Effect Size

