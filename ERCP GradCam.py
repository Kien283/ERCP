#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet18
from torchcam.methods import SmoothGradCAMpp
from torchvision.utils import save_image

from PIL import ImageOps


# In[73]:


root_path = Path("Split/train")
root_val = Path("Split/val")
root_test = Path("Split/test")

data_dir = os.path.join(root_path)
data_val = os.path.join(root_val)
data_test = os.path.join(root_test)


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


# In[76]:


val_frac = 0.0
test_frac = 0.0

length_val = (len(image_files_list_val))
length_test = (len(image_files_list_test))


indices_val = np.arange(length_val)
indices_test = np.arange(length_test)

val_x = [image_files_list_val[i] for i in indices_val]
val_y = [image_class_val[i] for i in indices_val]
test_x = [image_files_list_test[i] for i in indices_test]
test_y = [image_class_test[i] for i in indices_test]

print(
    f" Validation count: "
    f"{len(val_x)}, Test count: {len(test_x)}")


# In[77]:


train_transforms = Compose(
    [
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
        RandZoom(min_zoom=0.8, max_zoom=1.1, prob=0.5),
    ]
)

val_transforms = Compose(
    [LoadImage(image_only=True),  EnsureChannelFirst(), ScaleIntensity()])


y_pred_trans = Compose([Activations(softmax=True)])


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


# In[79]:


class ERCP(torch.utils.data.Dataset):
    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

val_ds = ERCP(val_x, val_y, val_transforms)
val_loader = DataLoader(
    val_ds, batch_size=160, num_workers=0)

test_ds = ERCP(test_x, test_y, val_transforms)
test_loader = DataLoader(
    test_ds, batch_size=100, num_workers=0)


# In[80]:


root_path = "Split/train"

model.load_state_dict(torch.load(
    os.path.join(root_path, "best_metric_model.pth")))


# In[81]:


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


# In[82]:


print("Test-group")
print(classification_report(
    y_true, y_pred, target_names=class_names_test, digits=4))


# In[83]:


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


fig, axis = plt.subplots(9, 9, figsize=(15,15))

x=0

for i in range (9):
    for j in range (9):
        axis[i][j].imshow(test_images[x].T, cmap="gray"); axis[i][j].axis('off')
        axis[i][j].set_title(f"Pred"+ str(y_pred[x]) +' Nr.'+ str(x))
        x=x+1
        if x == 70: 
            break

plt.savefig(f'Gradcam/Test')


# fig, axis = plt.subplots(13, 13, figsize=(15,15))
# 
# x=0
# 
# for i in range (13):
#     for j in range (13):
#         axis[i][j].imshow(val_images[x].T, cmap="gray")
#         axis[i][j].set_title(f"Pred"+ str(y_pred_val[x]) +' Nr.'+ str(x))
#         #axis[i][j].set_title(y_pred(x))
#         x=x+1
#         if x == 153: 
#             break
# 
# plt.savefig(f'Gradcam/Validation')
# 
# 

# In[ ]:


print(test_images.shape)
i=...
tens = torch.tensor(test_images[i]) 
print(tens.shape)
plt.imshow(tens.T, cmap="gray")


# In[ ]:


model = model.eval()

cam_extractor = SmoothGradCAMpp(model=model, input_shape=[1,224,224])

# Get your input

img = tens

input_tensor=tens

# Preprocess your data and feed it to the model
out = model(input_tensor.unsqueeze(0))

# Retrieve the CAM by passing the class index and the model output
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)


# In[ ]:


# Visualize the raw CAM
plt.imshow(activation_map[0].squeeze(0).numpy()); plt.axis('off'); plt.tight_layout(); plt.show()


# In[ ]:


import matplotlib.pyplot as plt

# Create a figure and an axis
fig, ax = plt.subplots()

# Display the activation map
im = ax.imshow(activation_map[0].squeeze(0), cmap="viridis")

# Add a colorbar to the figure
fig.colorbar(im)

# Hide the axis labels and ticks
ax.set_axis_off()

# Display the figure
plt.show()


# In[ ]:


from skimage.transform import resize

# Determine the size of the original image
original_size =tens.T.shape[:2]

# Resize the Grad-CAM visualization to the size of the original image
resized_cam = resize(activation_map[0].squeeze(0), original_size)

# Create a figure and an axis
fig, ax = plt.subplots()

# Display the activation map
im = ax.imshow(resized_cam.T, cmap="viridis")
ax.imshow(tens.T, cmap="gray", alpha=0.5); plt.axis('off'); plt.tight_layout();

# Add a colorbar to the figure

cbar = fig.colorbar(im)

# Hide the axis labels and ticks
ax.set_axis_off()

sm = cbar.mappable

# Set the scale of the colorbar to 0 and 1.0
sm.set_clim(0.0, 1.0)

cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
cbar.set_ticklabels([0, 0.25, 0.5, 0.75, 1.0])

# Display the figure
plt.show()


# img = read_image("rgb.png")
# 
# a=2
# 
# fig, axis = plt.subplots(a, a, figsize=(20,20)); plt.axis('off'); plt.tight_layout();
# 
# x=0
# 
# for i in range (a):
#    tens = torch.tensor(test_images[i])
# 
#    out = model(tens.unsqueeze(0))
#    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
# 
#    result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.0)
# 
#    result1 = ImageOps.mirror(result)
#    result1 = result1.rotate(90)
#    plt.figure(figsize = (5,5))
#    plt.imshow(img1, cmap="gray")
#    plt.imshow(result, alpha=0.5); plt.axis('off'); plt.tight_layout();
#    plt.savefig(f"overlay4"+str(i)+".png")
#    plt.show()
# 
# 

# img = read_image("rgb.png")
# 
# x=2
# 
# for i in range (0,90):
#     #model = model.eval()
#     cam_extractor = SmoothGradCAMpp(model=model, input_shape=[1,224,224])
#     tens = torch.tensor(test_images[i])
#     out = model(tens.unsqueeze(0))
#     activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)       
#     result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.0)
#     result1 = ImageOps.mirror(result)
#     result1 = result1.rotate(90)
#    
#     plt.figure("Cam", (12, 6))
#     plt.subplot(1, 3, 1)
#     plt.imshow(tens[0].T, cmap="gray"); plt.axis('off'); plt.tight_layout();
#     plt.subplot(1, 3, 2)
#     plt.imshow(result1); plt.axis('off'); plt.tight_layout();
#     plt.subplot(1, 3, 3)
#     plt.imshow(tens[0].T, cmap="gray")
#     plt.imshow(result1, alpha=0.5); plt.axis('off'); plt.tight_layout();
#     
#     plt.savefig(f'Gradcam/valAug'+str(i))
#     plt.show()
#     cam_extractor.remove_hooks()
#     

# In[84]:


## Create Grad-Cam for all 
from skimage.transform import resize

for i in range (0,152):
    #model = model.eval()
    cam_extractor = SmoothGradCAMpp(model=model, input_shape=[1,224,224])
    tens = torch.tensor(val_images[i])
    out = model(tens.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    
    original_size = tens.T.shape[:2]
    resized_cam = resize(activation_map[0].squeeze(0), original_size)
    cam = activation_map[0].squeeze(0)
   
    plt.figure("Cam", (12, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(tens[0].T, cmap="gray"); plt.axis('off'); plt.tight_layout();
    plt.subplot(1, 3, 2)
    plt.imshow(cam.T, cmap="viridis"); plt.axis('off'); plt.tight_layout();
    plt.subplot(1, 3, 3)

    im = plt.imshow(resized_cam.T, cmap="viridis", aspect="equal"); plt.axis('off'); plt.tight_layout();
    plt.imshow(tens[0].T, cmap="gray", alpha=0.5, aspect="equal")

    plt.savefig(f'Gradcam/val_1_'+str(i), bbox_inches="tight", dpi=600)
    plt.show()
    cam_extractor.remove_hooks()

