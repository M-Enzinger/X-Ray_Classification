import streamlit as st
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import semi_supervised
from PIL import Image
import glob
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

st.title('Team Chest X-Ray Classification')
st.write("Hello, we are Maxi and Ilayda. Our subject is **Chest X-Ray Classification**")

result_normal = st.button("Click here to see the x-ray image of a normal chest")
if result_normal:
    st.image('Normal-chest.jpeg', caption="this is what a normal chest looks like", width=None, channels="RGB",
             output_format="auto")

result_pneumonia = st.button("Click here to see the x-ray image of a chest with pneumonia")
if result_pneumonia:
    st.image('chest-pneumoia.jpeg', caption="this is what a chest with pneumonia looks like", width=None,
             channels="RGB", output_format="auto")

# Jan
labels = ['NORMAL', 'PNEUMONIA']
img_size = 2000


@st.cache
def get_data(data_dir):
    data = []
    for label in labels:
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data, dtype = object)


train = get_data('/Users/jan/Library/Mobile Documents/com~apple~CloudDocs/Uni/4. Semester/ML4B/Projekt/chest_xray/train')

val = get_data('/Users/jan/Library/Mobile Documents/com~apple~CloudDocs/Uni/4. Semester/ML4B/Projekt/chest_xray/test')

l = []
for i in train:
    if(i[1] == 0):
        l.append("NORMAL")
    else:
        l.append("PNEUMONIA")
sns.set_style('darkgrid')
sns.countplot(l)


