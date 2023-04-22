#!/usr/bin/env python
# coding: utf-8

# ## Setup environment

# ## Setup imports

# In[ ]:


# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import tempfile
import matplotlib.pyplot as plt
import PIL
import torch
import numpy as np
from sklearn.metrics import classification_report

from monai.apps import download_and_extract
from monai.config import print_config
from monai.data import decollate_batch, DataLoader
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    EnsureChannelFirst,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
)
from monai.utils import set_determinism

from pathlib import Path

from monai.networks.nets import resnet50

import torchvision

from sklearn import datasets, metrics, model_selection, svm


# ## Setup data directory

# In[100]:


root_path = Path("Split/train")
root_val = Path("Split/val")
root_test = Path("Split/test")


# In[ ]:


directory = os.environ.get("MONAI_DATA_DIRECTORY")
root_dir = tempfile.mkdtemp() if directory is None else directory
print(root_dir)


# In[102]:


data_dir = os.path.join(root_path)
data_val = os.path.join(root_val)
data_test = os.path.join(root_test)


# ## Read image filenames from the dataset folders

# In[ ]:


class_names_val = sorted(x for x in os.listdir(data_val)
                     if os.path.isdir(os.path.join(data_val, x)))
num_class_val = len(class_names_val)
image_files_val = [
    [
        os.path.join(data_val, class_names_val[i], x)
        for x in os.listdir(os.path.join(data_val, class_names_val[i]))
    ]
    for i in range(num_class_val)
]
num_each = [len(image_files_val[i]) for i in range(num_class_val)]
image_files_list_val = []
image_class_val = []
for i in range(num_class_val):
    image_files_list_val.extend(image_files_val[i])
    image_class_val.extend([i] * num_each[i])
num_total = len(image_class_val)
image_width, image_height = PIL.Image.open(image_files_list_val[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names_val}")
print(f"Label counts: {num_each}")


# In[ ]:


class_names_test = sorted(x for x in os.listdir(data_test)
                     if os.path.isdir(os.path.join(data_test, x)))
num_class_test = len(class_names_test)
image_files_test = [
    [
        os.path.join(data_test, class_names_test[i], x)
        for x in os.listdir(os.path.join(data_test, class_names_test[i]))
    ]
    for i in range(num_class_test)
]
num_each = [len(image_files_test[i]) for i in range(num_class_test)]
image_files_list_test = []
image_class_test = []
for i in range(num_class_test):
    image_files_list_test.extend(image_files_test[i])
    image_class_test.extend([i] * num_each[i])
num_total = len(image_class_test)
image_width, image_height = PIL.Image.open(image_files_list_test[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names_test}")
print(f"Label counts: {num_each}")


# In[ ]:


class_names = sorted(x for x in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, x)))
num_class = len(class_names)
image_files = [
    [
        os.path.join(data_dir, class_names[i], x)
        for x in os.listdir(os.path.join(data_dir, class_names[i]))
    ]
    for i in range(num_class)
]
num_each = [len(image_files[i]) for i in range(num_class)]
image_files_list = []
image_class = []
for i in range(num_class):
    image_files_list.extend(image_files[i])
    image_class.extend([i] * num_each[i])
num_total = len(image_class)
image_width, image_height = PIL.Image.open(image_files_list[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names}")
print(f"Label counts: {num_each}")


# ## Prepare training, validation and test data lists
# 

# In[ ]:


val_frac = 0.0
test_frac = 0.0
length = len(image_files_list)

length_val = (len(image_files_list_val))
length_test = (len(image_files_list_test))

indices = np.arange(length)
indices_val = np.arange(length_val)
indices_test = np.arange(length_test)

np.random.shuffle(indices)

test_split = int(test_frac * length)
val_split = int(val_frac * length) + test_split
test_indices = indices[:test_split]
train_indices = indices[val_split:]

train_x = [image_files_list[i] for i in train_indices]
train_y = [image_class[i] for i in train_indices]
val_x = [image_files_list_val[i] for i in indices_val]
val_y = [image_class_val[i] for i in indices_val]
test_x = [image_files_list_test[i] for i in indices_test]
test_y = [image_class_test[i] for i in indices_test]

print(
    f"Training count: {len(train_x)}, Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")


# ## Define MONAI transforms, Dataset and Dataloader to pre-process data

# In[108]:


train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),

    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True), EnsureChannelFirst(), ScaleIntensity()])

y_pred_trans = Compose([Activations(softmax=True)])
y_trans = Compose([AsDiscrete(to_onehot=num_class)])


# In[109]:


class ERCP(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_ds = ERCP(train_x, train_y, train_transforms)
train_loader = DataLoader(
    train_ds, batch_size=70, num_workers=0)

val_ds = ERCP(val_x, val_y, val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=200, num_workers=0)

test_ds = ERCP(test_x, test_y, val_transforms)
test_loader = DataLoader(
    test_ds, batch_size=100, num_workers=0)


# ## Define network and optimizer

# In[ ]:


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

model = DenseNet121(spatial_dims=2, in_channels=1,
                    out_channels=2).to(device)
                    
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
max_epochs = 100
val_interval = 1
auc_metric = ROCAUCMetric()
print(device)
torch.cuda.is_available()


# ## Evaluate the model on test dataset

# In[ ]:


model.load_state_dict(torch.load(
    os.path.join(root_path, "best_metric_model.pth")))

model.eval()

y_true = []
y_pred = []

with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
        test_data[0].to(device),
        test_data[1].to(device),
        )
        
        pred = model(test_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true.append(test_labels[i].item())
            y_pred.append(pred[i].item())

print(y_true)
print(y_pred)


# In[ ]:


print("Test-group")
print(classification_report(
    y_true, y_pred, target_names=class_names_test, digits=4))


# In[ ]:


pred = model(test_images)
ypred = pred[:,1]
ypred= ypred.detach().numpy()

fpr, tpr, threshold = metrics.roc_curve(y_true, ypred)
roc_auc = metrics.auc(fpr,tpr)
print(roc_auc)


# In[ ]:


plt.title('Test ROC')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("test_auroc.png")
plt.show()


# ## Evaluate the model on validation dataset

# In[ ]:


y_true_val = []
y_pred_val = []        
        
with torch.no_grad():
        for val_data in val_loader:
            val_images, val_labels = (
            val_data[0].to(device),
            val_data[1].to(device),
        )
        pred = model(val_images).argmax(dim=1)
        for i in range(len(pred)):
            y_true_val.append(val_labels[i].item())
            y_pred_val.append(pred[i].item())

print(y_true_val)
print(y_pred_val)

print(val_images.shape)
        


# In[ ]:


print("Validation-group")
print(classification_report(
    y_true_val, y_pred_val, target_names=class_names_test, digits=4))


# In[ ]:


pred_val = model(val_images)
ypred_val = pred_val[:,1]
ypred_val = ypred_val.detach().numpy()

fpr2, tpr2, threshold2 = metrics.roc_curve(y_true_val, ypred_val)
roc_auc = metrics.auc(fpr2,tpr2)
print(roc_auc)


# In[ ]:


plt.title('Validation ROC')
plt.plot(fpr2, tpr2, 'b', label = 'Run1: AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("Val_auroc.png")
plt.show()

