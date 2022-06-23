#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, models, transforms
import os
import torch.nn as nn

image_file = st.file_uploader("Upload An Image")
if image_file is not None:
    file_details = {"FileName":image_file.name,"FileType":image_file.type}
    st.write(file_details)
    img = load_image(image_file)
    st.image(img,height=250,width=250)
    with open(os.path.join("tempDir",image_file.name),"wb") as f: 
      f.write(image_file.getbuffer())         
    st.success("Saved File")
              
col1, col2, col3, col4, col5 = st.columns(5)
        with col3:
            button1 = st.button("Predict")
              

if (startButton):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # In[2]:


    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Directory of 'train' and 'test' folders
    data_dir = tempDir

    # Retrieving the images by folder
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'test']}
    class_names = image_datasets['train'].classes


    # In[3]:


    model = models.resnet18()
    num_ftrs = model.fc.in_features

    # Here the size of each output sample is set to 2.
    model.fc = nn.Linear(num_ftrs, 2)

    model.load_state_dict(torch.load("/models/FULLRetrainedResNetModel.pt"))

    model = model.to(device)


    # In[4]:


    number_of_predictions = 1

    with torch.no_grad():
        for data in dataloaders['test']:
            images, labels = data

            plt.imshow(np.squeeze(images[0][0:1]), cmap='gray')

            images = images.to(device)
            labels = labels.to(device)

            pred_outputs = model(images)
            predicted = torch.argmax(pred_outputs.data, 1)

    print('True Value: ', ' '.join('%s' % class_names[predicted[j]] for j in range(number_of_predictions)))
    print('Predicted: ', ' '.join('%s' % class_names[predicted[0]]
                                  for j in range(number_of_predictions)))


    # In[ ]:

    



