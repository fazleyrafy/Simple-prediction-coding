#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import pandas as pd
df = pd.DataFrame({
    'area':[2600, 3000, 3200, 3600, 4000],
    'price':[550000,565000,610000,680000,725000]
})
df


# In[2]:


plt.scatter(df.area, df.price)


# In[3]:


reg = linear_model.LinearRegression()


# In[4]:


reg.fit(df[['area']], df.price)


# In[5]:


reg.predict([[3600]])


# In[6]:


df


# In[10]:


reg.intercept_


# In[11]:


reg.coef_


# In[ ]:




