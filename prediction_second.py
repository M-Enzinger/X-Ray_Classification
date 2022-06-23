import streamlit as st
import os
import torch
from torch import nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

# Disable grad
with torch.no_grad():
  # Retrieve item
  index = 333
  item = '/img/Normal-chest.jpeg'
  image = item[0]
  true_target = item[1]

  # Loading the saved model
  save_path = '/models/FULLRetrainedResNetModel.pt'
  mlp = MLP()
  mlp.load_state_dict(torch.load(save_path))
  mlp.eval()

  # Generate prediction
  prediction = mlp(image)

  # Predicted class value using argmax
  predicted_class = np.argmax(prediction)

  # Reshape image
  image = image.reshape(600, 600, 1)

  # Show result
  st.write(plt.imshow(image, cmap='gray'))
  st.write(plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}'))
  st.write(plt.show())
