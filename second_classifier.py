import streamlit as st
import torchvision
import torch
import torchvision.transforms as transforms
import PIL.Image as Image
import pytorch_grad_cam

#image_path = "img/Normal-chest.jpeg"

classes = ["Pneumonia", "Normal"]

model = torch.load("models/FULLRetrainedResNetModel.pt", map_location=torch.device('cpu'))

mean = [0.4363, 0.4328, 0.3291]
std = [0.2129, 0.2075, 0.2038]

image_transforms = transforms.Compose([
  transforms.Resize((600, 600)),
  transforms.ToTenor(),
  transforms.Normalize(torch.Tenor(mean), torch.Tensor(std))
])

def classify(model, image_transforms, image_path, classes):
  model = model.eval()
  image = Image.open(image_path)
  image = image_transforms(image).float()
  image = image.unsqueeze(0)
  
  output = model(image)
  _, predicted = torch.max(output.data, 1)
  
  print(predicted.item())
  
st.write(classify(model, image_transforms, "img/Normal-chest.jpeg", classes))

                       
