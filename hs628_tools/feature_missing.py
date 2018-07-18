
# coding: utf-8

# In[2]:


import numpy as np


# In[ ]:


def __init__(self):
        
        # Dataframes recording information about features to remove
        self.missing_col = None
        self.single_unique = None
        self.collinear = None
        self.low_importance = None
        
        self.feature_importances = None
       
    


# In[34]:


a = np.array([[1,2,3], [4,5,6], [np.nan, np.nan, np.nan], [np.nan,8,np.nan]])


# In[35]:


a.shape[0]


# In[36]:


missing_entries = (len(a) - np.sum(a==a, axis=0)) / float(len(a))
missing_entries


# In[37]:


ms_th = 0.30
b= missing_entries > ms_th
missing_col = np.where(b)
missing_col[0]


# In[ ]:


def which_missing(self, data, missing_thresh):
       
       self.missing_thresh = missing_thresh

       # Calculate the fraction of missing in each column 
       missing_series = (len(a) - np.sum(data==data, axis=0)) / float(len(data))
       
       # Find the columns with a missing percentage above the threshold
       missing_col = np.where(missing_series > missing_thresh)

       missing_col = missing_col[0]

       self.missing_col = missing_col
       
       print('%d columns with greater than %0.2f missing values.\n' % (len(self.missing_col), self.missing_thresh))
       

