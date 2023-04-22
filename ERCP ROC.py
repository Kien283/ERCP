#!/usr/bin/env python
# coding: utf-8

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

import seaborn as sns

from sklearn.metrics import confusion_matrix

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import StratifiedKFold


# ## Setup data directory

# In[94]:


root_path = Path("Split/train")
root_val = Path("Split/val")
root_test = Path("Split/test")
root_external = Path("Split/external")


# In[ ]:


root_val1 = Path("Split1/val")
root_val2 = Path("Split2/val")
root_val3 = Path("Split3/val")

data_val1 = os.path.join(root_val1)
data_val2 = os.path.join(root_val2)
data_val3 = os.path.join(root_val3)


class_names_val1 = sorted(x for x in os.listdir(data_val1)
                     if os.path.isdir(os.path.join(data_val1, x)))
num_class_val1 = len(class_names_val1)
image_files_val1 = [
    [
        os.path.join(data_val1, class_names_val1[i], x)
        for x in os.listdir(os.path.join(data_val1, class_names_val1[i]))
    ]
    for i in range(num_class_val1)
]
num_each = [len(image_files_val1[i]) for i in range(num_class_val1)]
image_files_list_val1 = []
image_class_val1 = []
for i in range(num_class_val1):
    image_files_list_val1.extend(image_files_val1[i])
    image_class_val1.extend([i] * num_each[i])
num_total = len(image_class_val1)
image_width, image_height = PIL.Image.open(image_files_list_val1[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names_val1}")
print(f"Label counts: {num_each}")


class_names_val2 = sorted(x for x in os.listdir(data_val2)
                     if os.path.isdir(os.path.join(data_val2, x)))
num_class_val2 = len(class_names_val2)
image_files_val2 = [
    [
        os.path.join(data_val2, class_names_val2[i], x)
        for x in os.listdir(os.path.join(data_val2, class_names_val2[i]))
    ]
    for i in range(num_class_val2)
]
num_each = [len(image_files_val2[i]) for i in range(num_class_val2)]
image_files_list_val2 = []
image_class_val2 = []
for i in range(num_class_val2):
    image_files_list_val2.extend(image_files_val2[i])
    image_class_val2.extend([i] * num_each[i])
num_total = len(image_class_val2)
image_width, image_height = PIL.Image.open(image_files_list_val1[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names_val1}")
print(f"Label counts: {num_each}")


class_names_val3 = sorted(x for x in os.listdir(data_val3)
                     if os.path.isdir(os.path.join(data_val3, x)))
num_class_val3 = len(class_names_val3)
image_files_val3 = [
    [
        os.path.join(data_val3, class_names_val3[i], x)
        for x in os.listdir(os.path.join(data_val3, class_names_val3[i]))
    ]
    for i in range(num_class_val3)
]
num_each = [len(image_files_val3[i]) for i in range(num_class_val3)]
image_files_list_val3 = []
image_class_val3 = []
for i in range(num_class_val3):
    image_files_list_val3.extend(image_files_val3[i])
    image_class_val3.extend([i] * num_each[i])
num_total = len(image_class_val3)
image_width, image_height = PIL.Image.open(image_files_list_val1[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names_val1}")
print(f"Label counts: {num_each}")


# In[97]:


data_dir = os.path.join(root_path)
data_val = os.path.join(root_val)
data_test = os.path.join(root_test)
data_external = os.path.join(root_external)


# ## Read image filenames from the dataset folders

# In[99]:


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


# In[100]:


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


class_names_external = sorted(x for x in os.listdir(data_external)
                     if os.path.isdir(os.path.join(data_external, x)))
num_class_external = len(class_names_external)
image_files_external = [
    [
        os.path.join(data_external, class_names_external[i], x)
        for x in os.listdir(os.path.join(data_external, class_names_external[i]))
    ]
    for i in range(num_class_external)
]
num_each = [len(image_files_external[i]) for i in range(num_class_external)]
image_files_list_external = []
image_class_external = []
for i in range(num_class_external):
    image_files_list_external.extend(image_files_external[i])
    image_class_external.extend([i] * num_each[i])
num_total = len(image_class_external)
image_width, image_height = PIL.Image.open(image_files_list_external[0]).size

print(f"Total image count: {num_total}")
print(f"Image dimensions: {image_width} x {image_height}")
print(f"Label names: {class_names_external}")
print(f"Label counts: {num_each}")


# In[101]:


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

# In[102]:


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


# In[103]:


length_val1 = (len(image_files_list_val1))

indices_val1 = np.arange(length_val1)

val_x1 = [image_files_list_val1[i] for i in indices_val1]
val_y1 = [image_class_val1[i] for i in indices_val1]

length_val2 = (len(image_files_list_val2))

indices_val2 = np.arange(length_val2)

val_x2 = [image_files_list_val2[i] for i in indices_val2]
val_y2 = [image_class_val2[i] for i in indices_val2]

length_val3 = (len(image_files_list_val3))

indices_val3 = np.arange(length_val3)

val_x3 = [image_files_list_val3[i] for i in indices_val3]
val_y3 = [image_class_val3[i] for i in indices_val3]


# In[ ]:


length_external = (len(image_files_list_external))

indices_external = np.arange(length_external)

external_x = [image_files_list_external[i] for i in indices_external]
external_y = [image_class_external[i] for i in indices_external]


# ## Define MONAI transforms, Dataset and Dataloader to pre-process data

# In[104]:


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


# In[105]:


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
    train_ds, batch_size=60, shuffle=True, num_workers=0)

val_ds = ERCP(val_x, val_y, val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=160, num_workers=0)

test_ds = ERCP(test_x, test_y, val_transforms)
test_loader = DataLoader(
    test_ds, batch_size=100, num_workers=0)


# In[106]:


val_ds1 = ERCP(val_x1, val_y1, val_transforms)
val_loader1 = DataLoader(
    val_ds1, batch_size=160, num_workers=0)

val_ds2 = ERCP(val_x2, val_y2, val_transforms)
val_loader2 = DataLoader(
    val_ds2, batch_size=200, num_workers=0)

val_ds3 = ERCP(val_x3, val_y3, val_transforms)
val_loader3 = DataLoader(
    val_ds3, batch_size=200, num_workers=0)


# In[ ]:


external_ds = ERCP(external_x, external_y, val_transforms)
external_loader = DataLoader(
    external_ds, batch_size=100, num_workers=0)


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


# In[ ]:


y_true_test = []
y_pred_test = []

with torch.no_grad():
    for test_data in test_loader:
        test_images, test_labels = (
        test_data[0].to(device),
        test_data[1].to(device),
        )
        
        pred = model(test_images).argmax(dim=1)
        
        for i in range(len(pred)):
            y_true_test.append(test_labels[i].item())
            y_pred_test.append(pred[i].item())
            y_pred_test_prob.append(pred2[i])

print(y_true_test)
print(y_pred_test)


# ## Evaluate the model on validation dataset

# In[51]:


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
        


# In[110]:


y_true_val1 = []
y_pred_val1 = []        
        
with torch.no_grad():
        for val_data in val_loader1:
            val_images1, val_labels1 = (
            val_data[0].to(device),
            val_data[1].to(device),
        )
        pred = model(val_images1).argmax(dim=1)
        for i in range(len(pred)):
            y_true_val1.append(val_labels1[i].item())
            y_pred_val1.append(pred[i].item())

print(y_true_val1)
print(y_pred_val1)

print(val_images1.shape)

y_true_val2 = []
y_pred_val2 = []        
        
with torch.no_grad():
        for val_data in val_loader2:
            val_images2, val_labels2 = (
            val_data[0].to(device),
            val_data[1].to(device),
        )
        pred = model(val_images2).argmax(dim=1)
        for i in range(len(pred)):
            y_true_val2.append(val_labels2[i].item())
            y_pred_val2.append(pred[i].item())

print(y_true_val2)
print(y_pred_val2)

print(val_images2.shape)

y_true_val3 = []
y_pred_val3 = []        
        
with torch.no_grad():
        for val_data in val_loader3:
            val_images3, val_labels3 = (
            val_data[0].to(device),
            val_data[1].to(device),
        )
        pred = model(val_images3).argmax(dim=1)
        for i in range(len(pred)):
            y_true_val3.append(val_labels3[i].item())
            y_pred_val3.append(pred[i].item())

print(y_true_val3)
print(y_pred_val3)


# In[ ]:


y_true_external = []
y_pred_external = []

with torch.no_grad():
    for external_data in external_loader:
        external_images, external_labels = (
        test_data[0].to(device),
        test_data[1].to(device),
        )
        
        pred = model(external_images).argmax(dim=1)
        
        for i in range(len(pred)):
            y_true_external.append(external_labels[i].item())
            y_pred_external.append(pred[i].item())

print(y_true_external)
print(y_pred_external)


# In[116]:


###Cross valdiation

root_path="Split/#Models2"
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


for i in range(1, 4):
    model.load_state_dict(torch.load(
        os.path.join(root_path, f"best_metric_model"+str(i)+".pth")))
    model.eval()
    if i==1:
        y_score = model(val_images1)
        y_score = y_score.detach().numpy()
        fpr, tpr, _ = roc_curve(y_true_val1, y_score[:, 1])
    elif i==2:
        y_score = model(val_images2)
        y_score = y_score.detach().numpy()
        fpr, tpr, _ = roc_curve(y_true_val2, y_score[:, 1])
    elif i==3:
        y_score = model(val_images3)
        y_score = y_score.detach().numpy()
        fpr, tpr, _ = roc_curve(y_true_val3, y_score[:, 1])



    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC Split %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
 #                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Leipzig Valdiation cohort - cross-validation')
plt.legend(loc="lower right")
plt.savefig("Leipzig Valdiation cohort - cross-validation.png", dpi=600)
plt.show()


# In[114]:


#Cross-validation Test-set

root_path="Split/#Models2"
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


for i in range(1, 4):
    model.load_state_dict(torch.load(
        os.path.join(root_path, f"best_metric_model"+str(i)+".pth")))
    model.eval()
    y_score = model(test_images)
    y_score = y_score.detach().numpy()
    fpr, tpr, _ = roc_curve(y_true_test, y_score[:, 1])


    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC Split %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
 #                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Leipzig Test cohort - cross-validation')
plt.legend(loc="lower right")
plt.savefig("Leipzig Test cohort - cross-validation", dpi=600)
plt.show()


# In[ ]:


#external Datasets

root_path="Split/#Models"
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)


for i in range(1, 6):
    model.load_state_dict(torch.load(
        os.path.join(root_path, f"best_metric_model"+str(i)+".pth")))
    model.eval()
    y_score = model(external_images)
    y_score = y_score.detach().numpy()
    fpr, tpr, _ = roc_curve(y_true_external, y_score[:, 1])


    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC Split %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)
    
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
 #                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('external cohort')
plt.legend(loc="lower right")
plt.savefig("external cohort", dpi=600)
plt.show()

