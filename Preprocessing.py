#!/usr/bin/env python
# coding: utf-8

# In[30]:


import PIL
import os
import os.path
from PIL import Image, ImageEnhance
import cv2
from matplotlib import pyplot as plt


# In[27]:


def zoom_center(img, zoom_factor=1.6):

    y_size = img.shape[0]
    x_size = img.shape[1]
    
    # define new boundaries
    x1 = int(0.5*x_size*(1-1/zoom_factor))
    x2 = int(x_size-0.5*x_size*(1-1/zoom_factor))
    y1 = int(0.5*y_size*(1-1/zoom_factor))
    y2 = int(y_size-0.5*y_size*(1-1/zoom_factor))

    # first crop image then scale
    img_cropped = img[y1:y2,x1:x2]
    return cv2.resize(img_cropped, None, fx=zoom_factor, fy=zoom_factor)


# In[21]:


img_dir = r'/.../images/'


# In[ ]:


all_img = r'.../images/'
other_dir = r'.../images/'

for img in os.listdir(all_img):
    f_img = img_dir + img
    f, e = os.path.splitext(img_dir + img)
    img = cv2.imread(f_img,0)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20,20))
    
    equ = clahe.apply(img)
    
    cv2.imwrite(f+'.png', equ)
    
print('ende')


# In[ ]:


#Resize

print('Bulk images resizing started...')
for img in os.listdir(img_dir):
	f_img = img_dir + img
	f, e = os.path.splitext(img_dir + img)
	img = Image.open(f_img)
	img = img.resize((225, 225))
	img.save(f+ '.png')

print('Bulk images resizing finished...')


# In[ ]:


#Zoom

all_img = r'.../images/'
other_dir = r'.../images/'


for img in os.listdir(all_img):
    f_img = img_dir + img
    f, e = os.path.splitext(img_dir + img)
    img = cv2.imread(f_img,0)
    
    img_zoomed_and_cropped=zoom_center(img)
    
    cv2.imwrite(f+'.png', img_zoomed_and_cropped)
    
print('ende')

