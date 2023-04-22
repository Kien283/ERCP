#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os
pd.set_option('display.max_rows', None)


# In[55]:


from sklearn.model_selection import GroupShuffleSplit 

df = pd.read_csv('Label All.csv' , sep=';')

#Split in Train and Test
splitter = GroupShuffleSplit(test_size=.10, n_splits=2, random_state = 55)
split = splitter.split(df, groups=df['patient_id'])
train_inds, test_inds = next(split)

train = df.iloc[train_inds]
test = df.iloc[test_inds]


# In[ ]:


#Split Train in Train and Validation
df = train
splitter = GroupShuffleSplit(test_size=.15, n_splits=2, random_state = 55)
split = splitter.split(df, groups=df['patient_id'])
train_inds, val_inds = next(split)

train = df.iloc[train_inds]
val = df.iloc[val_inds]


# In[7]:


labels = train.sort_values('maligne')

for _, row in train.iterrows():
  f = row['id']
  l = row['maligne']
  os.replace(f'images/{f}'+'.png', f'Split/train/{l}/{f}'+'.png')


# In[8]:


labels = test.sort_values('maligne')

for _, row in test.iterrows():
  f = row['id']
  l = row['maligne']
  os.replace(f'images/{f}'+'.png', f'Split/test/{l}/{f}'+'.png')


# In[10]:


labels = val.sort_values('maligne')

for _, row in val.iterrows():
  f = row['id']
  l = row['maligne']
  os.replace(f'images/{f}'+'.png', f'Split/val/{l}/{f}'+'.png')


# In[63]:


#Cross-validation Split

from sklearn.model_selection import GroupShuffleSplit
import os
import shutil

df = pd.read_csv('label train and validation.csv' , sep=';')

# set the number of folds to 3
num_folds = 3

# create a GroupShuffleSplit object
splitter = GroupShuffleSplit(n_splits=num_folds, test_size=0.15, random_state=55)

# split the data into train and validation sets for each fold
for fold, (train_inds, val_inds) in enumerate(splitter.split(df, groups=df['patient_id'])):
    #print(f"Fold {fold}: Train indices={train_inds}, Validation indices={val_inds}")

    train = df.iloc[train_inds]
    val = df.iloc[val_inds]

    # loop over the train data rows
    for _, row in train.iterrows():
        f = row['id']
        l = row['maligne']
        src_path = os.path.join('images', f + '.png')
        dest_path = os.path.join('Split'+str(fold+1), 'train', str(l), f + '.png')
        shutil.copy(src_path, dest_path)

    # loop over the test data rows
    for _, row in val.iterrows():
        f = row['id']
        l = row['maligne']
        src_path = os.path.join('images', f + '.png')
        dest_path = os.path.join('Split'+str(fold+1), 'Val', str(l), f + '.png')
        shutil.copy(src_path, dest_path)

