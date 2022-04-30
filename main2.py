import streamlit as st
import pandas as pd
import numpy as np

st.title('X-Ray Classification Team - A Brief Introduction')
st.write("Hello, we are Maxi and Ilayda. Our subject is **Chest X-Ray Classification**")
st.balloons()

result_normal=st.button("Click here to see the x-ray image of a normal chest")
if result_normal:
  st.image('Normal-chest.jpeg', caption="this is what a normal chest looks like", width=None, channels="RGB", output_format="auto")
  
result_pneumonia=st.button("Click here to see the x-ray image of a chest with pneumonia")
if result_pneumonia:
  st.image('chest-pneumoia.jpeg', caption="this is what a chest with pneumonia looks like", width=None, channels="RGB", output_format="auto")
 
 
